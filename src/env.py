import gymnasium as gym
import mujoco
import numpy as np


class MorphologyAntEnv(gym.Wrapper):
    """
    Ant-v5 environment for morphology-robust intention-conditioned control.

    Intentions
    ----------
    0 = Forward
    1 = Left
    2 = Right
    3 = Stop

    Morphology parameters
    ---------------------
    mass:
        1.0 = nominal
        1.5 = +50%
        0.5 = -50%

    limb_length:
        1.0 = nominal
        1.5 = +50%
        0.5 = -50%

    joint_strength:
        1.0 = nominal
        0.5 = -50%

    friction:
        1.0 = nominal
        0.5 = -50%

    Observation
    -----------
    Standard Ant-v5 observation
        +
    4 morphology parameters
        +
    4-dimensional intention one-hot vector

    For Ant-v5 this is normally:

        105 + 4 + 4 = 113 dimensions

    Yaw / turning convention
    -------------------------
    The body frame used throughout this environment is
    x = forward, y = left, z = up (right-handed).

    get_yaw() computes atan2(rotation[3], rotation[0]), i.e. the
    angle of the forward axis in the world XY-plane using the
    standard atan2(y, x) convention. Under a right-handed frame
    with y = left, a POSITIVE yaw rotation turns the forward axis
    toward the left. Therefore:

        Left  intention -> positive yaw velocity target
        Right intention -> negative yaw velocity target

    (Previously these were swapped; see LEFT_YAW_TARGET /
    RIGHT_YAW_TARGET below.)
    """

    ACTION_DIM = 8
    INTENTION_DIM = 4

    INTENTIONS = {
        0: "Forward",
        1: "Left",
        2: "Right",
        3: "Stop",
    }

    # ------------------------------------------------------------------
    # Ant-v5 limb geometry names
    # ------------------------------------------------------------------

    LIMB_GEOMS = (
        "aux_1_geom",
        "left_leg_geom",
        "left_ankle_geom",
        "aux_2_geom",
        "right_leg_geom",
        "right_ankle_geom",
        "aux_3_geom",
        "back_leg_geom",
        "third_ankle_geom",
        "aux_4_geom",
        "rightback_leg_geom",
        "fourth_ankle_geom",
    )

    # Bodies whose local position determines the limb-chain geometry.
    LIMB_BODIES = (
        "aux_1",
        "left_leg",
        "aux_2",
        "right_leg",
        "aux_3",
        "back_leg",
        "aux_4",
        "rightback_leg",
    )

    # ------------------------------------------------------------------
    # Reward parameters
    # ------------------------------------------------------------------

    # Desired velocities.
    FORWARD_TARGET = 1.0

    # FIXED: these were previously swapped (LEFT was -0.8, RIGHT was
    # +0.8), which rewarded the policy for turning the OPPOSITE
    # direction of what "Left"/"Right" actually mean under this
    # environment's yaw convention (see class docstring above).
    LEFT_YAW_TARGET = 0.8
    RIGHT_YAW_TARGET = -0.8

    # Reward scales.
    FORWARD_TRACK_WEIGHT = 1.50
    FORWARD_LATERAL_WEIGHT = 0.50

    # FIXED: yaw_velocity is the only quantity affected by the dt bug
    # fixed in _step_dt() below (it used to be computed by dividing a
    # per-step yaw change by model.opt.timestep=0.01s, but one
    # env.step() actually advances the simulation by
    # frame_skip * timestep = 5 * 0.01 = 0.05s -- so yaw_velocity was
    # inflated 5x for the entire life of this environment).
    # forward_velocity / lateral_velocity come straight from MuJoCo's
    # qvel and were never affected by that bug.
    #
    # FORWARD_YAW_WEIGHT and STOP_YAW_WEIGHT are both applied as a
    # linear penalty directly on yaw_velocity's magnitude, so they
    # were implicitly tuned against the old, 5x-inflated scale. Once
    # the dt bug was fixed, the same physical spin rate produces 5x
    # LESS penalty than it effectively did before -- observed
    # concretely as Forward developing a +3.4 rad (~195 degree) yaw
    # drift and Stop drifting several meters, both with very high
    # energy use, once dt was corrected but these two weights were
    # left unchanged. Scaling both up 5x restores the original
    # relative anti-spin incentive now that yaw_velocity reflects the
    # true physical rate.
    #
    # TURN_YAW_WEIGHT is NOT rescaled: it is used as the exponential's
    # scale against an absolute target (LEFT_YAW_TARGET /
    # RIGHT_YAW_TARGET), not as a linear penalty on yaw_velocity's raw
    # magnitude, so fixing dt already correctly made that target
    # harder to satisfy (comparing against a true ~0.8 rad/s goal
    # instead of an accidentally-easy inflated one) without needing a
    # separate weight change.
    FORWARD_YAW_WEIGHT = 1.75  # was 0.35

    TURN_YAW_WEIGHT = 1.50
    TURN_FORWARD_WEIGHT = 0.20
    TURN_LATERAL_WEIGHT = 0.35

    STOP_LINEAR_WEIGHT = 1.20
    STOP_YAW_WEIGHT = 4.00  # was 0.80

    # Stability reward.
    UPRIGHT_WEIGHT = 0.30

    # Strong penalty after a custom fall.
    FALL_PENALTY = 5.0

    # Small action regularization term.
    CONTROL_COST_WEIGHT = 0.001

    # ------------------------------------------------------------------
    # Fall thresholds
    # ------------------------------------------------------------------

    FALL_HEIGHT = 0.25
    FALL_UPRIGHTNESS = 0.45

    RECOVERY_HEIGHT = 0.45
    RECOVERY_UPRIGHTNESS = 0.80

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        render_mode=None,
        morphology_observation=True,
        terminate_when_unhealthy=False,
    ):
        env = gym.make(
            "Ant-v5",
            render_mode=render_mode,
            terminate_when_unhealthy=terminate_when_unhealthy,
        )

        super().__init__(env)

        # FIXED: renamed from morphology_obs to morphology_observation
        # so this matches the keyword argument used by every calling
        # script (evaluate_intentions.py, run_experiments.py,
        # train_policy.py).
        self.morphology_observation = bool(
            morphology_observation
        )

        # --------------------------------------------------------------
        # Build the actual observation space.
        #
        # Gymnasium's Wrapper would otherwise retain Ant-v5's original
        # 105-dimensional observation space even though this wrapper
        # returns 113 dimensions.
        # --------------------------------------------------------------

        base_space = self.env.observation_space

        if not isinstance(
            base_space,
            gym.spaces.Box,
        ):
            raise TypeError(
                "Ant-v5 observation space must be a Box."
            )

        extra_dim = (
            4 if self.morphology_observation else 0
        ) + self.INTENTION_DIM

        low = np.concatenate(
            [
                np.asarray(
                    base_space.low,
                    dtype=np.float32,
                ),
                np.full(
                    extra_dim,
                    -np.inf,
                    dtype=np.float32,
                ),
            ]
        )

        high = np.concatenate(
            [
                np.asarray(
                    base_space.high,
                    dtype=np.float32,
                ),
                np.full(
                    extra_dim,
                    np.inf,
                    dtype=np.float32,
                ),
            ]
        )

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Action space
        # --------------------------------------------------------------

        if self.env.action_space.shape != (
            self.ACTION_DIM,
        ):
            raise ValueError(
                "Unexpected Ant-v5 action dimension: "
                f"{self.env.action_space.shape}"
            )

        # --------------------------------------------------------------
        # Intention
        # --------------------------------------------------------------

        self.intention = 0

        # --------------------------------------------------------------
        # Morphology
        # --------------------------------------------------------------

        self.current_params = np.ones(
            4,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        self.energy = 0.0

        self.initial_xy = np.zeros(
            2,
            dtype=np.float64,
        )

        self.initial_yaw = 0.0
        self._previous_yaw = 0.0

        self.fallen = False
        self.fall_step = None
        self.fall_start_time = None
        self.recovery_time = None

        self.step_count = 0

        # Most recent original Ant-v5 observation.
        self._last_base_obs = None

        # --------------------------------------------------------------
        # Store nominal MuJoCo model parameters.
        # --------------------------------------------------------------

        model = self.unwrapped.model

        self.nominal_body_mass = (
            model.body_mass.copy()
        )

        self.nominal_actuator_gear = (
            model.actuator_gear.copy()
        )

        self.nominal_geom_friction = (
            model.geom_friction.copy()
        )

        self.nominal_geom_size = (
            model.geom_size.copy()
        )

        self.nominal_body_pos = (
            model.body_pos.copy()
        )

        # --------------------------------------------------------------
        # Resolve limb IDs once.
        # --------------------------------------------------------------

        self.limb_geom_ids = []

        for name in self.LIMB_GEOMS:
            try:
                gid = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name,
                )

                if gid >= 0:
                    self.limb_geom_ids.append(
                        gid
                    )

            except Exception:
                pass

        self.limb_body_ids = []

        for name in self.LIMB_BODIES:
            try:
                bid = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )

                if bid >= 0:
                    self.limb_body_ids.append(
                        bid
                    )

            except Exception:
                pass

        if len(self.limb_geom_ids) == 0:
            raise RuntimeError(
                "No Ant limb geometries were found. "
                "Check the Ant-v5 geometry names."
            )

        if len(self.limb_body_ids) == 0:
            raise RuntimeError(
                "No Ant limb bodies were found. "
                "Check the Ant-v5 body names."
            )

    # ==================================================================
    # MORPHOLOGY
    # ==================================================================

    def _restore_nominal_morphology(self):
        """
        Restore all morphology-related MuJoCo model arrays to their
        original nominal values.
        """

        model = self.unwrapped.model

        model.body_mass[:] = (
            self.nominal_body_mass
        )

        model.actuator_gear[:] = (
            self.nominal_actuator_gear
        )

        model.geom_friction[:] = (
            self.nominal_geom_friction
        )

        model.geom_size[:] = (
            self.nominal_geom_size
        )

        model.body_pos[:] = (
            self.nominal_body_pos
        )

    def _scale_limb_geometry(
        self,
        limb_scale,
    ):
        """
        Scale Ant limb geometry and limb-chain body positions.

        Ant-v5 uses capsule geometries for the legs. For these
        geometries, geom_size[0] is the radius and geom_size[1]
        controls the capsule length.

        The radius is kept unchanged while the longitudinal
        dimension is scaled.
        """

        model = self.unwrapped.model

        if np.isclose(
            limb_scale,
            1.0,
        ):
            return

        # ----------------------------------------------------------
        # Scale limb body attachment positions.
        # ----------------------------------------------------------

        for bid in self.limb_body_ids:

            nominal_pos = (
                self.nominal_body_pos[bid]
            )

            model.body_pos[bid] = (
                nominal_pos * limb_scale
            )

        # ----------------------------------------------------------
        # Scale limb geometry.
        #
        # Ant-v5 leg geometries are capsules. Keep the radius
        # unchanged and scale the capsule half-length.
        # ----------------------------------------------------------

        for gid in self.limb_geom_ids:

            nominal_size = (
                self.nominal_geom_size[gid]
            )

            # Keep capsule radius unchanged.
            model.geom_size[
                gid,
                0,
            ] = nominal_size[0]

            # Scale capsule length.
            if model.geom_size.shape[1] > 1:

                model.geom_size[
                    gid,
                    1,
                ] = (
                    nominal_size[1]
                    * limb_scale
                )

        # ----------------------------------------------------------
        # Recompute MuJoCo dynamics after morphology changes.
        # ----------------------------------------------------------

        mujoco.mj_forward(
            model,
            self.unwrapped.data,
        )

    def _set_morphology(
        self,
        mass_scale=1.0,
        limb_scale=1.0,
        strength_scale=1.0,
        friction_scale=1.0,
    ):
        """
        Apply a complete morphology configuration.

        The model is always restored to nominal first, preventing
        perturbations from accumulating across experiments.
        """

        model = self.unwrapped.model

        # --------------------------------------------------------------
        # Always restore nominal values first.
        # --------------------------------------------------------------

        self._restore_nominal_morphology()

        # --------------------------------------------------------------
        # Validate parameters.
        # --------------------------------------------------------------

        scales = {
            "mass": mass_scale,
            "limb_length": limb_scale,
            "joint_strength": strength_scale,
            "friction": friction_scale,
        }

        for name, value in scales.items():

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        # --------------------------------------------------------------
        # 1. Global body mass.
        # --------------------------------------------------------------

        model.body_mass[:] = (
            self.nominal_body_mass
            * mass_scale
        )

        # --------------------------------------------------------------
        # 2. Actuator strength.
        # --------------------------------------------------------------

        model.actuator_gear[:] = (
            self.nominal_actuator_gear
            * strength_scale
        )

        # --------------------------------------------------------------
        # 3. Contact friction.
        # --------------------------------------------------------------

        model.geom_friction[:] = (
            self.nominal_geom_friction
            * friction_scale
        )

        # --------------------------------------------------------------
        # 4. Limb length.
        # --------------------------------------------------------------

        self._scale_limb_geometry(
            limb_scale
        )

        # --------------------------------------------------------------
        # Save current morphology.
        # --------------------------------------------------------------

        self.current_params = np.array(
            [
                mass_scale,
                limb_scale,
                strength_scale,
                friction_scale,
            ],
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Recompute MuJoCo forward dynamics after changing model
        # parameters.
        # --------------------------------------------------------------

        mujoco.mj_forward(
            model,
            self.unwrapped.data,
        )

    # ==================================================================
    # ORIENTATION
    # ==================================================================

    def _get_rotation_matrix(self):
        """
        Return the torso rotation matrix.
        """

        quat = np.asarray(
            self.unwrapped.data.qpos[3:7],
            dtype=np.float64,
        )

        rotation = np.zeros(
            9,
            dtype=np.float64,
        )

        mujoco.mju_quat2Mat(
            rotation,
            quat,
        )

        return rotation

    def _get_yaw(self):
        """
        Compute torso yaw from the torso quaternion.
        """

        rotation = (
            self._get_rotation_matrix()
        )

        return float(
            np.arctan2(
                rotation[3],
                rotation[0],
            )
        )

    @staticmethod
    def _wrap_angle(angle):
        """
        Wrap an angle to [-pi, pi].
        """

        return float(
            np.arctan2(
                np.sin(angle),
                np.cos(angle),
            )
        )

    def _get_uprightness(self):
        """
        Return torso uprightness.

        A value close to 1 means the torso's local z-axis points
        upward. Smaller values indicate increasing tilt.
        """

        rotation = (
            self._get_rotation_matrix()
        )

        return float(
            np.clip(
                rotation[8],
                -1.0,
                1.0,
            )
        )

    # ==================================================================
    # INTENTION
    # ==================================================================

    def set_intention(self, intention):
        """
        Set the current high-level motor intention.
        """

        intention = int(
            intention
        )

        if intention not in self.INTENTIONS:
            raise ValueError(
                f"Invalid intention {intention}. "
                f"Expected one of "
                f"{list(self.INTENTIONS.keys())}."
            )

        self.intention = intention

    def _intention_one_hot(self):
        """
        Return the intention as a one-hot vector.
        """

        one_hot = np.zeros(
            self.INTENTION_DIM,
            dtype=np.float32,
        )

        one_hot[
            self.intention
        ] = 1.0

        return one_hot

    # ==================================================================
    # OBSERVATION
    # ==================================================================

    def _build_observation(
        self,
        base_obs,
    ):
        """
        Build the complete policy observation.

        All training/evaluation code should use this function
        indirectly through get_current_observation().
        """

        parts = [
            np.asarray(
                base_obs,
                dtype=np.float32,
            )
        ]

        if self.morphology_observation:

            parts.append(
                self.current_params.astype(
                    np.float32
                )
            )

        parts.append(
            self._intention_one_hot()
        )

        observation = np.concatenate(
            parts
        ).astype(np.float32)

        return observation

    def get_current_observation(self):
        """
        Return the current complete observation.

        This is useful when the intention changes between physics
        steps, because the base Ant state does not need to be
        simulated again merely to change the intention vector.
        """

        if self._last_base_obs is None:
            raise RuntimeError(
                "No observation is available. "
                "Call reset() first."
            )

        return self._build_observation(
            self._last_base_obs
        )

    # ==================================================================
    # VELOCITY
    # ==================================================================

    def _velocity_components(
        self,
        yaw_velocity,
    ):
        """
        Compute torso-frame forward and lateral velocity.

        The Ant's qvel[0:2] gives translational velocity in the
        world frame. It is projected onto the torso's current
        horizontal forward and lateral directions.

        yaw_velocity is supplied separately from the actual
        quaternion yaw change.
        """

        data = self.unwrapped.data

        rotation = (
            self._get_rotation_matrix()
        )

        vx = float(
            data.qvel[0]
        )

        vy = float(
            data.qvel[1]
        )

        # First column of the rotation matrix:
        # torso forward direction in world coordinates.
        forward_x = rotation[0]
        forward_y = rotation[3]

        # Second column:
        # torso lateral direction in world coordinates.
        lateral_x = rotation[1]
        lateral_y = rotation[4]

        forward_velocity = (
            vx * forward_x
            + vy * forward_y
        )

        lateral_velocity = (
            vx * lateral_x
            + vy * lateral_y
        )

        return (
            float(forward_velocity),
            float(lateral_velocity),
            float(yaw_velocity),
        )

    # ==================================================================
    # REWARD
    # ==================================================================

    @staticmethod
    def _tracking_reward(
        error,
        scale,
    ):
        """
        Smooth bounded tracking reward.

        1.0 = exact target
        approaches 0.0 as error increases.
        """

        return float(
            np.exp(
                -scale * abs(error)
            )
        )

    def _intention_reward(
        self,
        forward_velocity,
        lateral_velocity,
        yaw_velocity,
        uprightness,
    ):
        """
        Compute the intention-specific reward.

        The reward explicitly distinguishes:

            Forward
            Left
            Right
            Stop

        while also penalizing unwanted motion and encouraging
        upright behavior.
        """

        if self.intention == 0:

            # ----------------------------------------------------------
            # Forward
            # ----------------------------------------------------------

            forward_tracking = (
                self._tracking_reward(
                    forward_velocity
                    - self.FORWARD_TARGET,
                    self.FORWARD_TRACK_WEIGHT,
                )
            )

            lateral_penalty = (
                abs(lateral_velocity)
                * self.FORWARD_LATERAL_WEIGHT
            )

            yaw_penalty = (
                abs(yaw_velocity)
                * self.FORWARD_YAW_WEIGHT
            )

            reward = (
                forward_tracking
                - lateral_penalty
                - yaw_penalty
            )

        elif self.intention == 1:

            # ----------------------------------------------------------
            # Left  (positive yaw target -- see class docstring)
            # ----------------------------------------------------------

            yaw_tracking = (
                self._tracking_reward(
                    yaw_velocity
                    - self.LEFT_YAW_TARGET,
                    self.TURN_YAW_WEIGHT,
                )
            )

            forward_penalty = (
                abs(forward_velocity)
                * self.TURN_FORWARD_WEIGHT
            )

            lateral_penalty = (
                abs(lateral_velocity)
                * self.TURN_LATERAL_WEIGHT
            )

            reward = (
                yaw_tracking
                - forward_penalty
                - lateral_penalty
            )

        elif self.intention == 2:

            # ----------------------------------------------------------
            # Right  (negative yaw target -- see class docstring)
            # ----------------------------------------------------------

            yaw_tracking = (
                self._tracking_reward(
                    yaw_velocity
                    - self.RIGHT_YAW_TARGET,
                    self.TURN_YAW_WEIGHT,
                )
            )

            forward_penalty = (
                abs(forward_velocity)
                * self.TURN_FORWARD_WEIGHT
            )

            lateral_penalty = (
                abs(lateral_velocity)
                * self.TURN_LATERAL_WEIGHT
            )

            reward = (
                yaw_tracking
                - forward_penalty
                - lateral_penalty
            )

        else:

            # ----------------------------------------------------------
            # Stop
            # ----------------------------------------------------------

            linear_speed = np.sqrt(
                forward_velocity ** 2
                + lateral_velocity ** 2
            )

            stop_penalty = (
                linear_speed
                * self.STOP_LINEAR_WEIGHT
                + abs(yaw_velocity)
                * self.STOP_YAW_WEIGHT
            )

            # Highest reward when all velocities approach zero.
            reward = np.exp(
                -stop_penalty
            )

        # --------------------------------------------------------------
        # Uprightness reward.
        #
        # This prevents the controller from obtaining an intention
        # reward by sacrificing physical stability.
        # --------------------------------------------------------------

        upright_reward = (
            np.clip(
                uprightness,
                0.0,
                1.0,
            )
            * self.UPRIGHT_WEIGHT
        )

        return float(
            reward + upright_reward
        )

    # ==================================================================
    # FALL DETECTION
    # ==================================================================

    def _detect_fall(self):
        """
        Detect a physical fall.

        A fall is declared if either:
            torso height < 0.25 m
        OR
            torso uprightness < 0.45
        """

        torso_height = float(
            self.unwrapped.data.qpos[2]
        )

        uprightness = (
            self._get_uprightness()
        )

        return bool(
            torso_height
            < self.FALL_HEIGHT
            or uprightness
            < self.FALL_UPRIGHTNESS
        )

    def _check_recovery(
        self,
        currently_fallen,
        current_time,
    ):
        """
        Record recovery time after a detected fall.

        Recovery requires both:
            torso height >= 0.45 m
            uprightness >= 0.80
        """

        if not self.fallen:
            return

        if self.recovery_time is not None:
            return

        if currently_fallen:
            return

        torso_height = float(
            self.unwrapped.data.qpos[2]
        )

        uprightness = (
            self._get_uprightness()
        )

        recovered = (
            torso_height
            >= self.RECOVERY_HEIGHT
            and uprightness
            >= self.RECOVERY_UPRIGHTNESS
        )

        if recovered:

            recovery_time = (
                current_time
                - self.fall_start_time
            )

            self.recovery_time = float(
                max(
                    0.0,
                    recovery_time,
                )
            )

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
        morphology=None,
        intention=0,
    ):
        """
        Reset the environment.

        Parameters
        ----------
        morphology:
            Either:

              - A dict containing any of:
                    mass, limb_length, joint_strength, friction
                (missing keys default to 1.0), OR

              - An array-like of exactly 4 values, in order:
                    [mass, limb_length, joint_strength, friction]

            If omitted, nominal morphology (all 1.0) is used.

            FIXED: previously this method only accepted a dict and
            called morphology.get(...), which raised an AttributeError
            whenever a caller passed a NumPy array (as every script in
            this project does). Both forms are now accepted.

        intention:
            0 = Forward
            1 = Left
            2 = Right
            3 = Stop
        """

        if morphology is None:
            morphology = np.ones(4, dtype=np.float32)

        if isinstance(morphology, dict):

            mass_scale = float(morphology.get("mass", 1.0))
            limb_scale = float(morphology.get("limb_length", 1.0))
            strength_scale = float(
                morphology.get("joint_strength", 1.0)
            )
            friction_scale = float(morphology.get("friction", 1.0))

        else:

            morphology_array = np.asarray(
                morphology,
                dtype=np.float32,
            ).reshape(-1)

            if morphology_array.shape[0] != 4:
                raise ValueError(
                    "morphology array must contain exactly 4 values "
                    "in order [mass, limb_length, joint_strength, "
                    f"friction]; got shape {morphology_array.shape}."
                )

            mass_scale = float(morphology_array[0])
            limb_scale = float(morphology_array[1])
            strength_scale = float(morphology_array[2])
            friction_scale = float(morphology_array[3])

        self.set_intention(
            intention
        )

        self._set_morphology(
            mass_scale=mass_scale,
            limb_scale=limb_scale,
            strength_scale=strength_scale,
            friction_scale=friction_scale,
        )

        # --------------------------------------------------------------
        # Reset the underlying Ant environment.
        # --------------------------------------------------------------

        base_obs, info = (
            self.env.reset(
                seed=seed,
                options=options,
            )
        )

        self._last_base_obs = (
            np.asarray(
                base_obs,
                dtype=np.float32,
            )
        )

        # --------------------------------------------------------------
        # Reset metrics.
        # --------------------------------------------------------------

        self.energy = 0.0

        self.step_count = 0

        self.fallen = False
        self.fall_step = None
        self.fall_start_time = None
        self.recovery_time = None

        # --------------------------------------------------------------
        # Initial position.
        # --------------------------------------------------------------

        self.initial_xy = (
            self.unwrapped.data.qpos[
                0:2
            ].copy()
        )

        # --------------------------------------------------------------
        # Initial orientation.
        # --------------------------------------------------------------

        self.initial_yaw = (
            self._get_yaw()
        )

        self._previous_yaw = (
            self.initial_yaw
        )

        return (
            self._build_observation(
                self._last_base_obs
            ),
            info,
        )

    # ==================================================================
    # STEP
    # ==================================================================

    def _step_dt(self):
        """
        Real simulated time that elapses during ONE env.step() call.

        Ant-v5 runs `frame_skip` (=5) physics sub-steps per env.step(),
        each of length model.opt.timestep (=0.01 s), so one env step is
        0.05 s -- NOT 0.01 s. env.unwrapped.dt already equals
        timestep * frame_skip, and is the correct value to use here.

        FIXED: this used to be model.opt.timestep directly (0.01 s).
        Dividing a per-step yaw change by 0.01 instead of the true
        0.05 inflated every reported/rewarded yaw_velocity by 5x for
        the entire life of this environment. Verified directly against
        the installed Ant-v5 model: env.unwrapped.dt == 0.05,
        env.unwrapped.model.opt.timestep == 0.01,
        env.unwrapped.frame_skip == 5.

        This does NOT affect yaw_change (computed via direct angle
        subtraction, no dt involved) or forward_velocity/
        lateral_velocity (read directly from MuJoCo's qvel) -- only
        yaw_velocity, and therefore the reward terms and energy/time
        bookkeeping that depend on it.
        """

        return float(self.unwrapped.dt)

    def step(self, action):
        """
        Execute one MuJoCo step and calculate custom metrics.
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Validate action.
        # --------------------------------------------------------------

        if action.shape != (
            self.ACTION_DIM,
        ):
            raise ValueError(
                "Expected action shape "
                f"({self.ACTION_DIM},), "
                f"got {action.shape}."
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Time step.
        # --------------------------------------------------------------

        dt = self._step_dt()

        previous_yaw = (
            self._previous_yaw
        )

        # --------------------------------------------------------------
        # Physics step.
        # --------------------------------------------------------------

        (
            base_obs,
            _base_reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            action
        )

        self._last_base_obs = (
            np.asarray(
                base_obs,
                dtype=np.float32,
            )
        )

        self.step_count += 1

        # --------------------------------------------------------------
        # Current yaw.
        # --------------------------------------------------------------

        current_yaw = (
            self._get_yaw()
        )

        yaw_delta_step = (
            self._wrap_angle(
                current_yaw
                - previous_yaw
            )
        )

        if dt > 0.0:

            yaw_velocity = (
                yaw_delta_step / dt
            )

        else:

            yaw_velocity = 0.0

        # --------------------------------------------------------------
        # Local linear velocities.
        # --------------------------------------------------------------

        (
            forward_velocity,
            lateral_velocity,
            yaw_velocity,
        ) = self._velocity_components(
            yaw_velocity
        )

        # --------------------------------------------------------------
        # Fall detection.
        # --------------------------------------------------------------

        currently_fallen = (
            self._detect_fall()
        )

        if (
            currently_fallen
            and not self.fallen
        ):

            self.fallen = True

            self.fall_step = (
                self.step_count
            )

            self.fall_start_time = (
                self.step_count * dt
            )

        # --------------------------------------------------------------
        # Recovery.
        # --------------------------------------------------------------

        current_time = (
            self.step_count * dt
        )

        self._check_recovery(
            currently_fallen,
            current_time,
        )

        # --------------------------------------------------------------
        # Energy.
        #
        # MuJoCo actuator generalized force x generalized velocity
        # gives an instantaneous mechanical power estimate.
        # --------------------------------------------------------------

        qfrc_actuator = (
            self.unwrapped.data.qfrc_actuator
        )

        qvel = (
            self.unwrapped.data.qvel
        )

        instantaneous_power = float(
            np.sum(
                np.abs(
                    qfrc_actuator
                    * qvel
                )
            )
        )

        self.energy += (
            instantaneous_power * dt
        )

        # --------------------------------------------------------------
        # Position.
        # --------------------------------------------------------------

        current_xy = (
            self.unwrapped.data.qpos[
                0:2
            ]
        )

        displacement = (
            current_xy
            - self.initial_xy
        )

        # Net displacement from starting point.
        net_distance = float(
            np.linalg.norm(
                displacement
            )
        )

        # Displacement projected onto the initial heading.
        initial_heading = np.array(
            [
                np.cos(
                    self.initial_yaw
                ),
                np.sin(
                    self.initial_yaw
                ),
            ],
            dtype=np.float64,
        )

        forward_distance = float(
            np.dot(
                displacement,
                initial_heading,
            )
        )

        # --------------------------------------------------------------
        # Episode yaw change.
        # --------------------------------------------------------------

        yaw_change = (
            self._wrap_angle(
                current_yaw
                - self.initial_yaw
            )
        )

        # --------------------------------------------------------------
        # Current stability.
        # --------------------------------------------------------------

        torso_height = float(
            self.unwrapped.data.qpos[2]
        )

        uprightness = (
            self._get_uprightness()
        )

        # --------------------------------------------------------------
        # Intention reward.
        # --------------------------------------------------------------

        intention_reward = (
            self._intention_reward(
                forward_velocity=forward_velocity,
                lateral_velocity=lateral_velocity,
                yaw_velocity=yaw_velocity,
                uprightness=uprightness,
            )
        )

        # --------------------------------------------------------------
        # Control cost.
        # --------------------------------------------------------------

        control_cost = (
            self.CONTROL_COST_WEIGHT
            * float(
                np.sum(
                    np.square(action)
                )
            )
        )

        reward = (
            intention_reward
            - control_cost
        )

        # --------------------------------------------------------------
        # Explicit fall penalty.
        # --------------------------------------------------------------

        if currently_fallen:

            reward -= (
                self.FALL_PENALTY
            )

        # --------------------------------------------------------------
        # Update yaw only after calculating this-step yaw velocity.
        # --------------------------------------------------------------

        self._previous_yaw = (
            current_yaw
        )

        # --------------------------------------------------------------
        # Energy efficiency.
        # --------------------------------------------------------------

        energy_per_distance = (
            self.energy
            / max(
                net_distance,
                1e-6,
            )
        )

        # --------------------------------------------------------------
        # Build information dictionary.
        # --------------------------------------------------------------

        info = dict(info)

        info.update(
            {
                # Position.
                "distance": float(
                    net_distance
                ),
                "net_distance": float(
                    net_distance
                ),
                "forward_distance": float(
                    forward_distance
                ),

                # Orientation.
                "yaw_change": float(
                    yaw_change
                ),
                "yaw_velocity": float(
                    yaw_velocity
                ),

                # Linear velocity.
                "forward_velocity": float(
                    forward_velocity
                ),
                "lateral_velocity": float(
                    lateral_velocity
                ),

                # Energy.
                "energy": float(
                    self.energy
                ),
                "energy_per_distance": float(
                    energy_per_distance
                ),

                # Intention.
                "intention": int(
                    self.intention
                ),
                "intention_name": (
                    self.INTENTIONS[
                        self.intention
                    ]
                ),

                # Stability.
                "fallen": bool(
                    self.fallen
                ),
                "currently_fallen": bool(
                    currently_fallen
                ),
                "fall_step": (
                    int(self.fall_step)
                    if self.fall_step is not None
                    else None
                ),
                "recovery_time": (
                    float(
                        self.recovery_time
                    )
                    if self.recovery_time is not None
                    else None
                ),
                "torso_height": float(
                    torso_height
                ),
                "uprightness": float(
                    uprightness
                ),

                # Episode bookkeeping.
                "steps": int(
                    self.step_count
                ),

                # Morphology.
                "morphology": (
                    self.current_params.copy()
                ),

                # Reward decomposition.
                "intention_reward": float(
                    intention_reward
                ),
                "control_cost": float(
                    control_cost
                ),
            }
        )

        # FIXED (design decision): terminate the episode the moment a
        # fall is detected, instead of letting the simulation continue
        # for the rest of the episode with the robot toppled over.
        #
        # Previously, self.fallen became True and stayed True, but
        # nothing set terminated=True, so a fallen robot kept being
        # simulated -- and kept receiving actuator commands from
        # whatever controller/policy was running -- for however many
        # steps were left. This produced two problems: (1) evaluation
        # scripts reading final-step info (distance, yaw_change,
        # energy) were measuring chaotic post-fall tumbling rather
        # than real walking behavior; and (2) FALL_PENALTY was being
        # applied every single step the robot stayed fallen, producing
        # enormous, hard-to-interpret negative episode rewards.
        #
        # Ending the episode on fall makes the reward signal cleaner
        # for both evaluation and PPO training, and matches standard
        # practice for legged-locomotion RL environments. The
        # RECOVERY_HEIGHT / RECOVERY_UPRIGHTNESS / recovery_time
        # machinery above is kept but is now effectively dead within a
        # single episode (recovery can never be observed if the
        # episode ends the instant a fall occurs) -- it's left in
        # place in case a future variant of this environment wants to
        # test recovery behavior instead of immediate termination.
        terminated = bool(terminated) or currently_fallen

        return (
            self._build_observation(
                self._last_base_obs
            ),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )