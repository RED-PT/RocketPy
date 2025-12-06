from rocketpy.motors import HybridMotor, MassFlowRateBasedTank
from .stochastic_motor_model import StochasticMotorModel
from .stochastic_tank import StochasticMassFlowRateBasedTank
from random import choice
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.mathutils.function import Function


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
    
    def dict_generator(self):
        """Special generator for the hybrid motor class that yields a dictionary
        with the randomly generated input arguments. This overrides the base
        dict_generator to exclude nested stochastic tank objects from being
        stored in last_rnd_dict, preventing JSON serialization issues.

        Yields
        ------
        dict
            Dictionary with the randomly generated input arguments.
        """
        generated_dict = next(super().dict_generator())
        # Replace tanks list with empty list to avoid storing StochasticTank objects
        generated_dict["tanks"] = []
        # Also clear the internal components map to avoid storing stochastic objects
        if "_StochasticHybridMotor__components_map" in generated_dict:
            generated_dict["_StochasticHybridMotor__components_map"] = {}
        self.last_rnd_dict = generated_dict
        yield generated_dict
    
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

        # Force recalculation of liquid motor's center_of_propellant_mass
        # by invalidating any cached versions so it gets recomputed with tanks
        if hasattr(hybrid_motor.liquid, '__dict__'):
            hybrid_motor.liquid.__dict__.pop('center_of_propellant_mass', None)

        # Wrap the liquid motor's center_of_propellant_mass to handle Function arithmetic
        def liquid_center_of_propellant_mass_wrapper(t=None):
            """Wrapper for liquid motor center of propellant mass that handles time arguments."""
            liquid = hybrid_motor.liquid
            total_mass = 0
            mass_balance = 0
            
            for positioned_tank in liquid.positioned_tanks:
                tank = positioned_tank.get("tank")
                tank_position = positioned_tank.get("position")
                
                if tank_position is None:
                    tank_position = 0
                
                # Get mass and center of mass, handling both Function and scalar types
                if hasattr(tank.fluid_mass, 'get_value_opt'):
                    tank_mass = tank.fluid_mass.get_value_opt(t) if t is not None else tank.fluid_mass.get_value_opt(0)
                else:
                    tank_mass = tank.fluid_mass
                
                if hasattr(tank.center_of_mass, 'get_value_opt'):
                    tank_com = tank.center_of_mass.get_value_opt(t) if t is not None else tank.center_of_mass.get_value_opt(0)
                else:
                    tank_com = tank.center_of_mass
                    
                if tank_mass is None:
                    tank_mass = 0
                if tank_com is None:
                    tank_com = 0
                    
                total_mass += tank_mass
                mass_balance += tank_mass * (tank_position + tank_com)
            
            return mass_balance / total_mass if total_mass > 0 else 0
        
        # Replace the liquid motor's center_of_propellant_mass with a Function wrapper
        hybrid_motor.liquid.__dict__['center_of_propellant_mass'] = Function(
            liquid_center_of_propellant_mass_wrapper,
            inputs="Time (s)",
            outputs="center of mass (m)"
        )

        # Wrap the HybridMotor's center_of_propellant_mass to handle Function arithmetic
        def hybrid_center_of_propellant_mass_wrapper(t=None):
            """Wrapper for hybrid motor center of propellant mass that handles time arguments."""
            solid = hybrid_motor.solid
            liquid = hybrid_motor.liquid
            
            # Get solid propellant mass and center of mass
            if hasattr(solid.propellant_mass, 'get_value_opt'):
                solid_mass = solid.propellant_mass.get_value_opt(t) if t is not None else solid.propellant_mass.get_value_opt(0)
            else:
                solid_mass = solid.propellant_mass
                
            if hasattr(solid.center_of_propellant_mass, 'get_value_opt'):
                solid_com = solid.center_of_propellant_mass.get_value_opt(t) if t is not None else solid.center_of_propellant_mass.get_value_opt(0)
            else:
                solid_com = solid.center_of_propellant_mass
            
            # Get liquid propellant mass
            if hasattr(liquid.propellant_mass, 'get_value_opt'):
                liquid_mass = liquid.propellant_mass.get_value_opt(t) if t is not None else liquid.propellant_mass.get_value_opt(0)
            else:
                liquid_mass = liquid.propellant_mass
            
            # Get liquid center of propellant mass using our wrapper
            if hasattr(liquid, '__dict__') and 'center_of_propellant_mass' in liquid.__dict__:
                liquid_com = liquid.__dict__['center_of_propellant_mass'].get_value_opt(t) if t is not None else liquid.__dict__['center_of_propellant_mass'].get_value_opt(0)
            else:
                liquid_com = 0
            
            # Get total propellant mass
            if hasattr(hybrid_motor.propellant_mass, 'get_value_opt'):
                total_mass = hybrid_motor.propellant_mass.get_value_opt(t) if t is not None else hybrid_motor.propellant_mass.get_value_opt(0)
            else:
                total_mass = hybrid_motor.propellant_mass
            
            mass_balance = solid_mass * solid_com + liquid_mass * liquid_com
            return mass_balance / total_mass if total_mass > 0 else 0
        
        # Replace the hybrid motor's center_of_propellant_mass with a Function wrapper
        hybrid_motor.__dict__['center_of_propellant_mass'] = Function(
            hybrid_center_of_propellant_mass_wrapper,
            inputs="Time (s)",
            outputs="center of mass (m)"
        )

        # Wrap all funcify_method decorated methods on the liquid motor to handle
        # Function arithmetic properly. This includes inertia tensors and other properties.
        def _create_liquid_motor_wrapper(motor_instance, method_name, original_method):
            """Create a wrapper for a liquid motor method that handles time arguments."""
            def wrapper(t=None):
                """Wrapper that safely calls the original method and handles Function objects."""
                try:
                    # Try calling the original method
                    result = original_method(motor_instance)
                    
                    # If result is a Function, evaluate it at time t
                    if hasattr(result, 'get_value_opt'):
                        return result.get_value_opt(t) if t is not None else result.get_value_opt(0)
                    return result
                except (TypeError, AttributeError):
                    # Fallback: return None or 0
                    return 0
            
            return wrapper

        # List of methods to wrap on the liquid motor
        liquid_motor_methods_to_wrap = [
            'propellant_I_11',
            'propellant_I_22', 
            'propellant_I_33',
            'propellant_I_12',
            'propellant_I_13',
            'propellant_I_23',
        ]
        
        for method_name in liquid_motor_methods_to_wrap:
            if hasattr(hybrid_motor.liquid.__class__, method_name):
                original_method = getattr(hybrid_motor.liquid.__class__, method_name)
                # Get the underlying function from the funcify_method descriptor
                if hasattr(original_method, 'func'):
                    orig_func = original_method.func
                    wrapper_func = _create_liquid_motor_wrapper(hybrid_motor.liquid, method_name, orig_func)
                    hybrid_motor.liquid.__dict__[method_name] = Function(
                        wrapper_func,
                        inputs="Time (s)",
                        outputs="Inertia (kg m²)"
                    )

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
        except Exception as e:
            # Don't let the sanitizer crash stochastic creation; it's a
            # best-effort compatibility layer.
            import warnings
            warnings.warn(f"Sanitizer failed during stochastic motor creation: {e}")

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



