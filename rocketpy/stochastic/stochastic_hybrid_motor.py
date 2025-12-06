"""
Defines the StochasticHybridMotor class for Monte Carlo simulation of hybrid
rocket motors with uncertainty quantification.

This module provides stochastic wrappers for hybrid motors that can have
randomized parameters for Monte Carlo analysis. It properly handles dynamic
tank addition and uses Function arithmetic for complex step differentiation
compatibility during flight simulation.
"""

from random import choice

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.motors import HybridMotor, MassFlowRateBasedTank

from .stochastic_motor_model import StochasticMotorModel
from .stochastic_tank import StochasticMassFlowRateBasedTank


class StochasticHybridMotor(StochasticMotorModel):
    """A Stochastic Hybrid Motor class that inherits from StochasticMotorModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`HybridMotor <rocketpy.motors.HybridMotor>`

    Attributes
    ----------
    object : HybridMotor
        HybridMotor object to be used for validation.
    thrust_source : int, float, tuple, list
        Thrust source file path or list of paths.
    total_impulse : int, float, tuple, list
        Total impulse of the motor in Ns.
    burn_start_time : int, float, tuple, list
        Burn start time of the motor in seconds.
    burn_out_time : int, float, tuple, list
        Burn out time of the motor in seconds.
    dry_mass : int, float, tuple, list
        Dry mass of the motor in kg.
    dry_I_11 : int, float, tuple, list
        Moment of inertia in the x direction in kg*m^2.
    dry_I_22 : int, float, tuple, list
        Moment of inertia in the y direction in kg*m^2.
    dry_I_33 : int, float, tuple, list
        Moment of inertia in the z direction in kg*m^2.
    dry_I_12 : int, float, tuple, list
        Product of inertia in kg*m^2.
    dry_I_13 : int, float, tuple, list
        Product of inertia in kg*m^2.
    dry_I_23 : int, float, tuple, list
        Product of inertia in kg*m^2.
    nozzle_radius : int, float, tuple, list
        Nozzle radius of the motor in meters.
    grain_number : int, float, tuple, list
        Number of grains in the motor.
    grain_density : int, float, tuple, list
        Density of the grain in kg/m^3.
    grain_outer_radius : int, float, tuple, list
        Outer radius of the grain in meters.
    grain_initial_inner_radius : int, float, tuple, list
        Initial inner radius of the grain in meters.
    grain_initial_height : int, float, tuple, list
        Initial height of the grain in meters.
    grain_separation : int, float, tuple, list
        Separation between grains in meters.
    grains_center_of_mass_position : int, float, tuple, list
        Position of the center of mass of the grains in meters.
    center_of_dry_mass_position : int, float, tuple, list
        Position of the center of dry mass in meters.
    nozzle_position : int, float, tuple, list
        Position of the nozzle in meters.
    throat_radius : int, float, tuple, list
        Throat radius of the motor in meters.
    tanks : list
        List of StochasticMassFlowRateBasedTank objects. This cannot be
        randomized directly but tanks can be added via add_tank method.
    """

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
        throat_radius=None,
    ):
        """Initializes the Stochastic Hybrid Motor class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        hybrid_motor : HybridMotor
            HybridMotor object to be used for validation.
        thrust_source : int, float, tuple, list, optional
            Thrust source file path or list of paths.
        total_impulse : int, float, tuple, list, optional
            Total impulse of the motor in Ns.
        burn_start_time : int, float, tuple, list, optional
            Burn start time of the motor in seconds.
        burn_out_time : int, float, tuple, list, optional
            Burn out time of the motor in seconds.
        dry_mass : int, float, tuple, list, optional
            Dry mass of the motor in kg.
        dry_inertia_11 : int, float, tuple, list, optional
            Moment of inertia in the x direction in kg*m^2.
        dry_inertia_22 : int, float, tuple, list, optional
            Moment of inertia in the y direction in kg*m^2.
        dry_inertia_33 : int, float, tuple, list, optional
            Moment of inertia in the z direction in kg*m^2.
        dry_inertia_12 : int, float, tuple, list, optional
            Product of inertia in kg*m^2.
        dry_inertia_13 : int, float, tuple, list, optional
            Product of inertia in kg*m^2.
        dry_inertia_23 : int, float, tuple, list, optional
            Product of inertia in kg*m^2.
        nozzle_radius : int, float, tuple, list, optional
            Nozzle radius of the motor in meters.
        grain_number : int, float, tuple, list, optional
            Number of grains in the motor.
        grain_density : int, float, tuple, list, optional
            Density of the grain in kg/m^3.
        grain_outer_radius : int, float, tuple, list, optional
            Outer radius of the grain in meters.
        grain_initial_inner_radius : int, float, tuple, list, optional
            Initial inner radius of the grain in meters.
        grain_initial_height : int, float, tuple, list, optional
            Initial height of the grain in meters.
        grain_separation : int, float, tuple, list, optional
            Separation between grains in meters.
        grains_center_of_mass_position : int, float, tuple, list, optional
            Position of the center of mass of the grains in meters.
        center_of_dry_mass_position : int, float, tuple, list, optional
            Position of the center of dry mass in meters.
        nozzle_position : int, float, tuple, list, optional
            Position of the nozzle in meters.
        throat_radius : int, float, tuple, list, optional
            Throat radius of the motor in meters.
        """
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
    
    def _set_stochastic(self, seed=None):
        """Set the stochastic attributes for tanks and motor inputs.

        This method is called when resetting the stochastic structure of the
        motor, including all nested tank components.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator.
        """
        super()._set_stochastic(seed)
        self.tanks = self._reset_tanks(seed)
    
    def dict_generator(self):
        """Special generator for the hybrid motor class that yields a
        dictionary with the randomly generated input arguments. This overrides
        the base dict_generator to exclude nested stochastic tank objects from
        being stored in last_rnd_dict, preventing JSON serialization issues.
        
        Note: The tanks list is initialized as empty here and populated in
        create_object() after each tank is generated. This ensures we store
        the final randomized parameters rather than the stochastic objects.

        Yields
        ------
        dict
            Dictionary with the randomly generated input arguments.
        """
        generated_dict = next(super().dict_generator())
        # Initialize tanks list as empty - it will be populated in create_object
        # with each tank's last_rnd_dict and position after creation
        generated_dict["tanks"] = []
        # Also clear the internal components map to avoid storing stochastic
        # objects (which are not JSON serializable)
        if "_StochasticHybridMotor__components_map" in generated_dict:
            generated_dict["_StochasticHybridMotor__components_map"] = {}
        self.last_rnd_dict = generated_dict
        yield generated_dict

    def _reset_tanks(self, seed=None):
        """Reset the stochastic structure of all tanks in this motor.

        This method resets the internal state of all stochastic tanks,
        allowing them to generate new randomized parameters with an
        optional seed.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator.

        Returns
        -------
        new_tanks : list
            A list of dictionaries containing the reset tanks and their
            positions.
        """
        new_tanks = []
        for tank_entry in self.tanks:
            stochastic_tank = tank_entry["tank"]
            tank_position = tank_entry["position"]
            
            # Reset the stochastic structure of the tank
            stochastic_tank._set_stochastic(seed)
            
            # Re-validate the position after reset
            validated_position = self._validate_position(stochastic_tank, tank_position)
            
            new_tanks.append({"tank": stochastic_tank, "position": validated_position})
        
        return new_tanks

    @staticmethod
    def _create_liquid_center_of_mass_function(liquid_motor):
        """Create a Function for liquid motor's center of propellant mass using
        Function arithmetic to support complex step differentiation.
        
        This manual wrapping is necessary because tanks are added dynamically
        to the liquid motor after initialization. The standard funcify_method
        decorator cannot automatically handle this dynamic composition, so we
        manually compute the weighted center of mass from all positioned tanks.
        
        This implementation uses Function arithmetic (addition, multiplication,
        division) rather than lambda evaluation to ensure complex step
        differentiation works correctly during flight simulation.

        Parameters
        ----------
        liquid_motor : LiquidMotor
            The liquid motor instance whose center of mass needs wrapping.

        Returns
        -------
        Function
            Function object that computes center of propellant mass as a
            weighted average of all tank contributions. Uses Function arithmetic
            to support complex step differentiation.
        """
        # Get fallback position in case of empty tanks
        fallback_position = (
            liquid_motor.positioned_tanks[0].get("position", 0) 
            if liquid_motor.positioned_tanks else 0
        )
        
        def compute_center_of_mass(t):
            """Compute weighted center of mass from all tanks at time t.
            
            This approach evaluates each tank's properties at the given time,
            allowing complex step differentiation to work correctly while
            avoiding division by zero and NaN issues.
            """
            total_mass = 0
            mass_balance = 0
            
            for positioned_tank in liquid_motor.positioned_tanks:
                tank = positioned_tank.get("tank")
                tank_position = positioned_tank.get("position", 0)
                
                # Evaluate tank mass and center of mass at time t
                if hasattr(tank.fluid_mass, '__call__'):
                    tank_mass = tank.fluid_mass(t)
                else:
                    tank_mass = tank.fluid_mass
                    
                if hasattr(tank.center_of_mass, '__call__'):
                    tank_com = tank.center_of_mass(t)
                else:
                    tank_com = tank.center_of_mass
                
                # Accumulate
                if np.isfinite(tank_mass) and np.isfinite(tank_com):
                    total_mass += tank_mass
                    mass_balance += tank_mass * (tank_position + tank_com)
            
            # Safe division with fallback
            if total_mass == 0 or not np.isfinite(total_mass):
                return fallback_position
            result = mass_balance / total_mass
            return result if np.isfinite(result) else fallback_position
            
        return Function(compute_center_of_mass)

    @staticmethod
    def _create_hybrid_center_of_mass_function(hybrid_motor):
        """Create a Function for hybrid motor's center of propellant mass using
        Function arithmetic to support complex step differentiation.
        
        This manual wrapping is necessary for the same reason as the liquid
        center of mass - the hybrid motor composition is dynamic and includes
        tanks added after initialization. We compute a weighted center of mass
        from both the solid grain and all liquid tanks.
        
        This implementation mirrors the base HybridMotor class's approach,
        using direct evaluation at each time point for robust handling of edge
        cases while supporting complex step differentiation.

        Parameters
        ----------
        hybrid_motor : HybridMotor
            The hybrid motor instance whose center of mass needs wrapping.

        Returns
        -------
        Function
            Function object that computes combined center of propellant mass
            from solid and liquid components.
        """
        solid = hybrid_motor.solid
        liquid = hybrid_motor.liquid
        
        # Get fallback position (center of dry mass)
        fallback_position = hybrid_motor.center_of_dry_mass_position

        def compute_center_of_mass(t):
            """Compute weighted center of mass from solid and liquid components.
            
            This approach evaluates each component's properties at the given time,
            allowing complex step differentiation to work correctly while
            avoiding division by zero and NaN issues.
            """
            # Get solid propellant mass and center of mass
            if hasattr(solid.propellant_mass, '__call__'):
                solid_mass = solid.propellant_mass(t)
            else:
                solid_mass = solid.propellant_mass
                
            if hasattr(solid.center_of_propellant_mass, '__call__'):
                solid_com = solid.center_of_propellant_mass(t)
            else:
                solid_com = solid.center_of_propellant_mass
            
            # Get liquid propellant mass  
            if hasattr(liquid.propellant_mass, '__call__'):
                liquid_mass = liquid.propellant_mass(t)
            else:
                liquid_mass = liquid.propellant_mass
            
            # Get liquid center of mass from wrapped function if it exists
            if hasattr(liquid, '__dict__') and 'center_of_propellant_mass' in liquid.__dict__:
                liquid_com_func = liquid.__dict__['center_of_propellant_mass']
                liquid_com = liquid_com_func(t) if hasattr(liquid_com_func, '__call__') else liquid_com_func
            else:
                # Fallback to the property
                if hasattr(liquid.center_of_propellant_mass, '__call__'):
                    liquid_com = liquid.center_of_propellant_mass(t)
                else:
                    liquid_com = liquid.center_of_propellant_mass
            
            # Compute mass balance
            if not (np.isfinite(solid_mass) and np.isfinite(solid_com) and 
                    np.isfinite(liquid_mass) and np.isfinite(liquid_com)):
                return fallback_position
                
            mass_balance = solid_mass * solid_com + liquid_mass * liquid_com
            total_mass = solid_mass + liquid_mass
            
            # Safe division with fallback
            if total_mass == 0 or not np.isfinite(total_mass):
                return fallback_position
            result = mass_balance / total_mass
            return result if np.isfinite(result) else fallback_position
            
        return Function(compute_center_of_mass)
    def create_object(self):
        """Creates and returns a HybridMotor object from the randomly
        generated input arguments.

        Returns
        -------
        HybridMotor
            HybridMotor object with the randomly generated input arguments.
        """
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
            burn_time=(
                generated_dict["burn_start_time"],
                generated_dict["burn_out_time"],
            ),
            throat_radius=generated_dict["throat_radius"],
            reshape_thrust_curve=(
                (generated_dict["burn_start_time"], generated_dict["burn_out_time"]),
                generated_dict["total_impulse"],
            ),
            coordinate_system_orientation=generated_dict[
                "coordinate_system_orientation"
            ],
            interpolation_method=generated_dict["interpolate"],
        )

        # Add all tanks to the motor
        for tank_entry in self.tanks:
            tank_obj = tank_entry["tank"]
            tank, position_rnd = self._create_tank(tank_obj)
            hybrid_motor.add_tank(tank, position_rnd)

        # Wrap liquid motor's center_of_propellant_mass
        if hasattr(hybrid_motor, "liquid") and hybrid_motor.liquid is not None:
            # Clear any cached center_of_propellant_mass
            if hasattr(hybrid_motor.liquid, "__dict__"):
                hybrid_motor.liquid.__dict__.pop("center_of_propellant_mass", None)

            hybrid_motor.liquid.__dict__["center_of_propellant_mass"] = (
                self._create_liquid_center_of_mass_function(hybrid_motor.liquid)
            )

        # Wrap hybrid motor's center_of_propellant_mass
        hybrid_motor.__dict__["center_of_propellant_mass"] = (
            self._create_hybrid_center_of_mass_function(hybrid_motor)
        )

        # Note: The center of mass wrapping above is necessary because tanks
        # are added dynamically. RocketPy's reset_funcified_methods will not
        # automatically recalculate these. The wrapping ensures proper
        # time-dependent evaluation of tank contributions.

        return hybrid_motor

    def add_tank(self, tank, position):
        """Add a tank to the stochastic hybrid motor.

        Parameters
        ----------
        tank : StochasticMassFlowRateBasedTank, MassFlowRateBasedTank
            Tank object to be added to the motor. If a MassFlowRateBasedTank
            is provided, it will be converted to a StochasticMassFlowRateBasedTank
            automatically.
        position : int, float, tuple, list
            Position of the tank in meters relative to the motor's coordinate
            system origin.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If tank is not of type MassFlowRateBasedTank or
            StochasticMassFlowRateBasedTank.
        """

        if not isinstance(
            tank, (StochasticMassFlowRateBasedTank, MassFlowRateBasedTank)
        ):
            raise AssertionError(
                "`tank` must be of MassFlowRateBasedTank or "
                "StochasticMassFlowRateBasedTank type"
            )
        if isinstance(tank, MassFlowRateBasedTank):
            tank = StochasticMassFlowRateBasedTank(
                mass_flow_rate_based_tank=tank
            )
        
        # Validate and store position
        validated_position = self._validate_position(tank, position)
        
        # Store the provided position on the stochastic tank object so that
        # downstream code (e.g. _create_tank) can read it via
        # `stochastic_tank.position`.
        try:
            tank.position = validated_position
        except Exception:
            # If for some reason the tank object is not writable, fall back
            # to keeping the position only in the internal lists.
            pass
        
        self.__components_map[tank] = validated_position
        self.tanks.append({"tank": tank, "position": validated_position})

    def _randomize_position(self, position):
        """Randomize a position provided as a tuple or list.
        
        Parameters
        ----------
        position : tuple, list, int, float
            Position to be randomized.
            
        Returns
        -------
        int, float
            Randomized position value.
        """
        if isinstance(position, tuple):
            if isinstance(position[0], Vector):
                # TODO: implement randomization for X and Y positions
                return position[-1](position[0].z, position[1])
            return position[-1](position[0], position[1])
        elif isinstance(position, list):
            return choice(position) if position else position
        return position

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
        """Create a tank object from a stochastic tank and store its
        randomized parameters in the parent's last_rnd_dict.
        
        Parameters
        ----------
        stochastic_tank : StochasticMassFlowRateBasedTank
            Stochastic tank object to create a tank from.
            
        Returns
        -------
        tuple
            Tuple containing the created tank and its randomized position.
        """
        tank = stochastic_tank.create_object()
        position_rnd = self._randomize_position(stochastic_tank.position)
        
        # Store the tank's randomized parameters and position in the parent's
        # last_rnd_dict for complete traceability (similar to StochasticRocket)
        self.last_rnd_dict["tanks"].append(stochastic_tank.last_rnd_dict.copy())
        self.last_rnd_dict["tanks"][-1]["position"] = position_rnd
        
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
        # Try to get position from object
        error_msg = (
            "`position` standard deviation was provided but the motor does "
            f"not have the same {validated_object.obj.__class__.__name__} "
            "to get the nominal position value from."
        )

        if isinstance(validated_object, StochasticMassFlowRateBasedTank):
            if isinstance(validated_object.obj, MassFlowRateBasedTank):

                def get_tank_position(self_object, _):
                    # For HybridMotor, tanks are stored in the liquid motor
                    liquid_motor = getattr(self_object, 'liquid', self_object)
                    positioned_tanks = getattr(liquid_motor, 'positioned_tanks', [])
                    
                    for tank_dict in positioned_tanks:
                        tank = tank_dict.get("tank")
                        position = tank_dict.get("position")
                        if tank == validated_object.obj:
                            return position
                    raise AssertionError(error_msg)

                return get_tank_position



