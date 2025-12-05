from rocketpy.motors import HybridMotor, MassFlowRateBasedTank
from .stochastic_motor_model import StochasticMotorModel
from .stochastic_tank import StochasticMassFlowRateBasedTank
from random import choice
from rocketpy.mathutils.vector_matrix import Vector


class StochasticHybridMotor(StochasticMotorModel):

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        hybrid_motor,
        thrust_source=None,
        total_impulse=None,
        burn_start_time=None,
        burn_out_time=None,
        dry_mass=None,
        dry_inertia_11=None,
        dry_inertia_22=None,
        dry_inertia_33=None,
        dry_inertia_12=None,
        dry_inertia_13=None,
        dry_inertia_23=None,
        nozzle_radius=None,
        grain_number=None,
        grain_density=None,
        grain_outer_radius=None,
        grain_initial_inner_radius=None,
        grain_initial_height=None,
        grain_separation=None,
        grains_center_of_mass_position=None,
        center_of_dry_mass_position=None,
        nozzle_position=None,
        throat_radius=None
    ):
        self.tanks = []
        self.__components_map = {}

        super().__init__(
            hybrid_motor,
            thrust_source=thrust_source,
            total_impulse=total_impulse,
            burn_start_time=burn_start_time,
            burn_out_time=burn_out_time,
            dry_mass=dry_mass,
            dry_I_11=dry_inertia_11,
            dry_I_22=dry_inertia_22,
            dry_I_33=dry_inertia_33,
            dry_I_12=dry_inertia_12,
            dry_I_13=dry_inertia_13,
            dry_I_23=dry_inertia_23,
            nozzle_radius=nozzle_radius,
            grain_number=None,
            grain_density=grain_density,
            grain_outer_radius=grain_outer_radius,
            grain_initial_inner_radius=grain_initial_inner_radius,
            grain_initial_height=grain_initial_height,
            grain_separation=grain_separation,
            grains_center_of_mass_position=grains_center_of_mass_position,
            center_of_dry_mass_position=center_of_dry_mass_position,
            nozzle_position=nozzle_position,
            throat_radius=throat_radius,
            interpolate=None,
            coordinate_system_orientation=None
        )
    
    def create_object(self):

        generated_dict = next(self.dict_generator())
        
        hybrid_motor = HybridMotor(
            thrust_source=generated_dict["thrust_source"],
            dry_mass=generated_dict["dry_mass"],
            dry_inertia=(
                generated_dict["dry_I_11"],
                generated_dict["dry_I_22"],
                generated_dict["dry_I_33"],
                generated_dict["dry_I_12"],
                generated_dict["dry_I_13"],
                generated_dict["dry_I_23"],
            ),
            nozzle_radius=generated_dict["nozzle_radius"],
            grain_number=generated_dict["grain_number"],
            grain_density=generated_dict["grain_density"],
            grain_outer_radius=generated_dict["grain_outer_radius"],
            grain_initial_inner_radius=generated_dict["grain_initial_inner_radius"],
            grain_initial_height=generated_dict["grain_initial_height"],
            grain_separation=generated_dict["grain_separation"],
            grains_center_of_mass_position=generated_dict[
                "grains_center_of_mass_position"
            ], 
            center_of_dry_mass_position=generated_dict["center_of_dry_mass_position"],
            nozzle_position=generated_dict["nozzle_position"],
            burn_time = (
                generated_dict["burn_start_time"], generated_dict["burn_out_time"]
            ),
            throat_radius=generated_dict["throat_radius"],
            reshape_thrust_curve=(
                (generated_dict["burn_start_time"], generated_dict["burn_out_time"]),
                generated_dict["total_impulse"],
            ),
            coordinate_system_orientation=generated_dict["coordinate_system_orientation"],
            interpolation_method=generated_dict["interpolate"],
        )
        
        for tank_entry in self.tanks:
            tank_obj = tank_entry["tank"]
            tank, position_rnd = self._create_tank(tank_obj)
            hybrid_motor.add_tank(tank, position_rnd)

        # Sanitize funcified methods on the created motor and its subcomponents
        # so that their Function sources robustly accept a time argument.
        # This avoids changing the base LiquidMotor implementation while
        # ensuring the stochastic creation returns objects whose
        # funcified attributes behave correctly when used in arithmetic
        # or evaluated by the Function machinery.
        try:
            from rocketpy.mathutils.function import Function as _RFunction

            def _sanitize_instance_funcs(inst):
                """Replace funcified descriptors on inst with safe Function
                wrappers that call the original decorated method in a way
                that tolerates both signatures (with or without time).
                """
                if inst is None:
                    return
                cls = type(inst)
                for name, attr in cls.__dict__.items():
                    # funcify_method decorator instances store original func
                    # under the attribute `func` on the descriptor.
                    if hasattr(attr, "func"):
                        orig = attr.func

                        def make_source(orig_func, instance):
                            def source(*args, **kwargs):
                                # Prefer calling as orig(instance, t) when the
                                # method expects a time argument. Fallback to
                                # calling orig(instance) and adapting its
                                # result (Function or callable) to return a
                                # value for the provided args.
                                try:
                                    # Try direct call with args (most common)
                                    return orig_func(instance, *args, **kwargs)
                                except TypeError:
                                    # No-arg method: call without args and
                                    # interpret returned value.
                                    res = orig_func(instance)
                                    # If it returned a Function-like object
                                    # use its callable interface.
                                    try:
                                        if hasattr(res, "get_value_opt"):
                                            # Function instance
                                            if args:
                                                return res.get_value_opt(args[0])
                                            return res
                                        if callable(res):
                                            return res(*args, **kwargs)
                                    except Exception:
                                        # Give up and re-raise original TypeError
                                        raise
                                    return res

                            return source

                        # Create a new Function wrapper and cache it on the
                        # instance dictionary so subsequent accesses use it.
                        try:
                            inst.__dict__[name] = _RFunction(
                                make_source(orig, inst), inputs="Time (s)"
                            )
                        except Exception:
                            # Best-effort: ignore failures to avoid breaking
                            # stochastic creation; the original object will
                            # be returned in that case.
                            pass

            _sanitize_instance_funcs(hybrid_motor)
            _sanitize_instance_funcs(getattr(hybrid_motor, "liquid", None))
            _sanitize_instance_funcs(getattr(hybrid_motor, "solid", None))
        except Exception:
            # Don't let the sanitizer crash stochastic creation; it's a
            # best-effort compatibility layer.
            pass

        return hybrid_motor


    def add_tank(self, tank, position):

        if not isinstance(tank, (StochasticMassFlowRateBasedTank, MassFlowRateBasedTank)):
            raise AssertionError(
                "`tank` must be of MassFlowRateBasedtank or StochasticMassFlowRateBasedtank type"
            )
        if isinstance(tank, MassFlowRateBasedTank):
            tank = StochasticMassFlowRateBasedTank(mass_flow_rate_based_tank=tank)
        # store the provided position on the stochastic tank object so that
        # downstream code (e.g. _create_tank) can read it via
        # `stochastic_tank.position`.
        try:
            tank.position = position
        except Exception:
            # If for some reason the tank object is not writable, fall back
            # to keeping the position only in the internal lists.
            pass
        self.__components_map[tank] = position
        self.tanks.append({"tank": tank, "position": position})

    def _randomize_position(self, position):
        """Randomize a position provided as a tuple or list."""
        if isinstance(position, tuple):
            if isinstance(position[0], Vector):
                # TODO implement randomization for X and Y positions
                return position[-1](position[0].z, position[1])
            return position[-1](position[0], position[1])
        elif isinstance(position, list):
            return choice(position) if position else position
        
    def _validate_position(self, validated_object, position):
        """Validate the position argument.

        Parameters
        ----------
        validated_object : object
            The object to which the position argument refers to.
        position : tuple, list, int, float
            The position argument to be validated.

        Returns
        -------
        tuple or list
            Validated position argument.

        Raises
        ------
        ValueError
            If the position argument does not conform to the specified formats.
        """
        if isinstance(position, tuple):
            return self._validate_tuple(
                "position",
                position,
                getattr=self._create_get_position(validated_object),
            )
        
        elif isinstance(position, (int, float)):
            return self._validate_scalar(
                "position",
                position,
                getattr=self._create_get_position(validated_object),
            )
        elif isinstance(position, list):
            return self._validate_list(
                "position",
                position,
                getattr=self._create_get_position(validated_object),
            )
        elif position is None:
            position = []
            return self._validate_list(
                "position",
                position,
                getattr=self._create_get_position(validated_object),
            )
        else:
            raise AssertionError("`position` must be a tuple, list, int, or float")

    def _create_tank(self, stochastic_tank):
        tank = stochastic_tank.create_object()
        position_rnd = self._randomize_position(
           stochastic_tank.position
        )
        return tank, position_rnd
    
    def _create_get_position(self, validated_object):
        """Create a function to get the nominal position from an object.

        Parameters
        ----------
        validated_object : object
            The object to which the position argument refers to.

        Returns
        -------
        function
            Function to get the nominal position from an object. The function
            must receive two arguments.
        """

        # try to get position from object
        error_msg = (
            "`position` standard deviation was provided but the motor does "
            f"not have the same {validated_object.obj.__class__.__name__} "
            "to get the nominal position value from."
        )

        if isinstance(validated_object, StochasticMassFlowRateBasedTank):
            if isinstance(validated_object.obj, MassFlowRateBasedTank):
                for tank, position in self.__components_map.items():
                    
                    def get_tank_position(self_object, _):

                        for tank, position in self_object.positioned_tanks:
                            if tank == validated_object.obj:
                                return position
                        raise AssertionError(error_msg)
                    
                    return  get_tank_position



