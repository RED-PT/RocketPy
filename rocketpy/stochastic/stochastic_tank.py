"""
Defines the StochasticCylindricalTank and StochasticMassFlowRateBasedTank classes
for Monte Carlo simulation of rocket propulsion tanks with uncertainty quantification.

This module provides stochastic wrappers for tank classes that enable randomization
of tank geometry and mass flow parameters. It supports property-based nominal value
inheritance and includes robust validation with automatic retry logic for parameter
compatibility.
"""

from rocketpy.motors.tank import MassFlowRateBasedTank
from rocketpy.motors.tank_geometry import CylindricalTank

from .stochastic_model import StochasticModel


class StochasticCylindricalTank(StochasticModel):
    """A Stochastic Cylindrical Tank class that inherits from StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`CylindricalTank <rocketpy.motors.tank.CylindricalTank>`

    Attributes
    ----------
    object : CylindricalTank
        CylindricalTank object to be used for validation.
    input_radius : int, float, tuple, list
        Radius of the tank in meters. Nominal value inherited from
        cylindrical_tank.input_radius property.
    input_height : int, float, tuple, list
        Height of the tank in meters. Nominal value inherited from
        cylindrical_tank.input_height property.
    spherical_caps : bool
        Whether the tank has spherical caps. This cannot be randomized.
    """

    def __init__(
        self,
        cylindrical_tank,
        radius=None,
        height=None,
    ):
        """Initializes the Stochastic Cylindrical Tank class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        cylindrical_tank : CylindricalTank
            CylindricalTank object to be used for validation.
        radius : int, float, tuple, list, optional
            Radius of the tank in meters. If None, uses the nominal value
            from cylindrical_tank.input_radius property.
        height : int, float, tuple, list, optional
            Height of the tank in meters. If None, uses the nominal value
            from cylindrical_tank.input_height property.
        """
        self.spherical_caps = cylindrical_tank.has_caps
        super().__init__(
            cylindrical_tank,
            input_radius=radius,
            input_height=height,
        )

    def create_object(self, max_attempts=100):
        """Creates and returns a CylindricalTank object from the randomly generated
        input arguments. If validation fails, retries with new random parameters.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum number of attempts to generate a valid tank configuration.
            Default is 100.

        Returns
        -------
        CylindricalTank
            CylindricalTank object with the randomly generated input arguments.

        Raises
        ------
        ValueError
            If unable to generate a valid tank configuration after max_attempts
            tries. This indicates the parameter ranges are likely incompatible.
        """
        for attempt in range(max_attempts):
            generated_dict = next(self.dict_generator())

            try:
                tank_geometry = CylindricalTank(
                    radius=generated_dict["input_radius"],
                    height=generated_dict["input_height"],
                    spherical_caps=self.spherical_caps,
                )
                # If successful, return the valid tank
                return tank_geometry
            except ValueError as e:
                # Log the failed attempt (only on last attempt)
                if attempt == max_attempts - 1:
                    error_msg = (
                        f"\n{'='*70}\n"
                        f"Validation Error in StochasticCylindricalTank\n"
                        f"{'='*70}\n"
                        f"Failed to generate valid tank after {max_attempts} attempts.\n"
                        f"Last error: {str(e)}\n\n"
                        f"Last attempted parameters:\n"
                        f"  - Radius: {generated_dict['input_radius']:.3f} m\n"
                        f"  - Height: {generated_dict['input_height']:.3f} m\n"
                        f"  - Spherical caps: {self.spherical_caps}\n\n"
                        f"Suggestion: Adjust the stochastic parameter ranges to ensure\n"
                        f"positive values for radius and height.\n"
                        f"{'='*70}"
                    )
                    raise ValueError(error_msg) from e
                # Otherwise, continue to next attempt
                continue

        # This should never be reached due to the exception above
        raise ValueError("Unexpected error in tank generation")


class StochasticMassFlowRateBasedTank(StochasticModel):
    """A Stochastic Mass Flow Rate Based Tank class that inherits from
    StochasticModel.

    See Also
    --------
    :ref:`stochastic_model` and :class:`MassFlowRateBasedTank <rocketpy.motors.tank.MassFlowRateBasedTank>`

    Attributes
    ----------
    object : MassFlowRateBasedTank
        MassFlowRateBasedTank object to be used for validation.
    geometry : StochasticCylindricalTank, CylindricalTank
        Geometry of the tank. Can be a stochastic or static geometry object.
    flux_start_time : int, float, tuple, list
        Time in seconds when the flux starts.
    flux_stop_time : int, float, tuple, list
        Time in seconds when the flux stops.
    liquid : Fluid, list
        Liquid fluid object or list of fluid objects.
    gas : Fluid, list
        Gas fluid object or list of fluid objects.
    initial_liquid_mass : int, float, tuple, list
        Initial mass of liquid in kg.
    initial_gas_mass : int, float, tuple, list
        Initial mass of gas in kg.
    total_liquid_mass_flow : int, float, tuple, list
        Total liquid mass consumed in kg. Used with reshape_mass_flow_curve.
    total_gas_mass_flow : int, float, tuple, list
        Total gas mass consumed in kg. Used with reshape_mass_flow_curve.
    name : str, list
        Name of the tank. This cannot be randomized.
    liquid_mass_flow_rate_in : str, list
        Path to CSV file with liquid mass flow rate in. This cannot be randomized.
    liquid_mass_flow_rate_out : str, list
        Path to CSV file with liquid mass flow rate out. This cannot be randomized.
    gas_mass_flow_rate_in : str, list
        Path to CSV file with gas mass flow rate in. This cannot be randomized.
    gas_mass_flow_rate_out : str, list
        Path to CSV file with gas mass flow rate out. This cannot be randomized.
    """

    def __init__(
        self,
        mass_flow_rate_based_tank,
        geometry,
        flux_start_time=None,
        flux_stop_time=None,
        liquid=None,
        gas=None,
        initial_liquid_mass=None,
        initial_gas_mass=None,
        total_liquid_mass_flow=None,
        total_gas_mass_flow=None,
    ):
        """Initializes the Stochastic Mass Flow Rate Based Tank class.

        See Also
        --------
        :ref:`stochastic_model`

        Parameters
        ----------
        mass_flow_rate_based_tank : MassFlowRateBasedTank
            MassFlowRateBasedTank object to be used for validation.
        geometry : StochasticCylindricalTank, CylindricalTank
            Geometry of the tank. Can be a stochastic or static geometry object.
        flux_start_time : int, float, tuple, list, optional
            Time in seconds when the flux starts.
        flux_stop_time : int, float, tuple, list, optional
            Time in seconds when the flux stops.
        liquid : Fluid, list, optional
            Liquid fluid object or list of fluid objects.
        gas : Fluid, list, optional
            Gas fluid object or list of fluid objects.
        initial_liquid_mass : int, float, tuple, list, optional
            Initial mass of liquid in kg.
        initial_gas_mass : int, float, tuple, list, optional
            Initial mass of gas in kg.
        total_liquid_mass_flow : int, float, tuple, list, optional
            Total liquid mass consumed (outflow - inflow) in kg over the 
            flux time period. Used to reshape the liquid mass flow curves.
        total_gas_mass_flow : int, float, tuple, list, optional
            Total gas mass consumed (outflow - inflow) in kg over the 
            flux time period. Used to reshape the gas mass flow curves.
        """
        self.geometry = geometry

        super().__init__(
            mass_flow_rate_based_tank,
            flux_start_time=flux_start_time,
            flux_stop_time=flux_stop_time,
            liquid=liquid,
            gas=gas,
            initial_liquid_mass=initial_liquid_mass,
            initial_gas_mass=initial_gas_mass,
            total_liquid_mass_flow=total_liquid_mass_flow,
            total_gas_mass_flow=total_gas_mass_flow,
            gas_mass_flow_rate_in=None,
            gas_mass_flow_rate_out=None,
            liquid_mass_flow_rate_in=None,
            liquid_mass_flow_rate_out=None,
            name=None,
        )

    @staticmethod
    def _is_stochastic_geometry(geometry):
        """Check if the geometry is a stochastic object.

        Parameters
        ----------
        geometry : object
            Geometry object to check.

        Returns
        -------
        bool
            True if geometry is a StochasticModel, False otherwise.
        """
        return isinstance(geometry, StochasticModel)

    @staticmethod
    def _reshape_mass_flow_curve(
        mass_flow_in, mass_flow_out, new_flux_time, total_mass_flow
    ):
        """Reshape mass flow curves to match new flux time and total mass flow.
        
        This method reshapes the mass flow rate curves (in and out) to align 
        with a new flux time period and achieve a target total mass consumed.
        The shape of the curves is preserved, but they are scaled in time and 
        magnitude.
        
        Parameters
        ----------
        mass_flow_in : Function
            Original mass flow rate into the tank in kg/s.
        mass_flow_out : Function
            Original mass flow rate out of the tank in kg/s.
        new_flux_time : tuple of float
            New flux time period (start_time, stop_time) in seconds.
        total_mass_flow : float
            Target total mass consumed (outflow - inflow) in kg over the 
            flux time period.
            
        Returns
        -------
        tuple
            Tuple containing (reshaped_mass_flow_in, reshaped_mass_flow_out),
            both as Function objects.
            
        Notes
        -----
        The reshaping process:
        1. Time scaling: The original time array is scaled to fit the new 
           flux time period.
        2. Magnitude scaling: Both curves are scaled proportionally to achieve 
           the target total mass flow.
        3. The net mass flow (outflow - inflow) is preserved in shape but 
           scaled to match the target.
        """
        from rocketpy.mathutils.function import Function

        # Get original data arrays
        time_in, flow_in = mass_flow_in.x_array, mass_flow_in.y_array
        time_out, flow_out = mass_flow_out.x_array, mass_flow_out.y_array

        # Get original flux time bounds
        old_flux_start = min(time_in[0], time_out[0])
        old_flux_stop = max(time_in[-1], time_out[-1])
        old_flux_duration = old_flux_stop - old_flux_start

        # Calculate time scaling factor
        new_flux_start, new_flux_stop = new_flux_time
        new_flux_duration = new_flux_stop - new_flux_start
        time_scale = new_flux_duration / old_flux_duration

        # Scale time arrays
        new_time_in = (time_in - old_flux_start) * time_scale + new_flux_start
        new_time_out = (time_out - old_flux_start) * time_scale + new_flux_start

        # Create temporary functions with scaled time
        temp_flow_in = Function(
            source=list(zip(new_time_in, flow_in)),
            inputs="Time (s)",
            outputs="Mass Flow Rate (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )
        temp_flow_out = Function(
            source=list(zip(new_time_out, flow_out)),
            inputs="Time (s)",
            outputs="Mass Flow Rate (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )

        # Calculate old total mass flow (net outflow - inflow)
        # We need to integrate the net flow over the new time period
        old_total_mass_flow = (
            temp_flow_out.integral(new_flux_start, new_flux_stop)
            - temp_flow_in.integral(new_flux_start, new_flux_stop)
        )

        # Avoid division by zero
        if abs(old_total_mass_flow) < 1e-10:
            raise ValueError(
                "Cannot reshape mass flow curves: original net mass flow is zero. "
                "Check that the original mass flow curves have non-zero net flow."
            )

        # Calculate magnitude scaling factor
        magnitude_scale = total_mass_flow / old_total_mass_flow

        # Apply magnitude scaling to flow rates
        new_flow_in = flow_in * magnitude_scale
        new_flow_out = flow_out * magnitude_scale

        # Create reshaped functions
        reshaped_flow_in = Function(
            source=list(zip(new_time_in, new_flow_in)),
            inputs="Time (s)",
            outputs="Mass Flow Rate In (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )
        reshaped_flow_out = Function(
            source=list(zip(new_time_out, new_flow_out)),
            inputs="Time (s)",
            outputs="Mass Flow Rate Out (kg/s)",
            interpolation="linear",
            extrapolation="zero",
        )

        return reshaped_flow_in, reshaped_flow_out


    def _set_stochastic(self, seed=None):
        """Set the stochastic attributes for geometry and tank inputs.

        This method is called when resetting the stochastic structure of the
        tank, including any nested stochastic geometry. This ensures that when
        a parent StochasticHybridMotor resets its state, all child tanks are
        also reset with the same seed for reproducibility.

        Note: total_liquid_mass_flow and total_gas_mass_flow are now properties
        of the base MassFlowRateBasedTank class. The stochastic parameters with
        these names are used to reshape the mass flow curves to achieve target
        total mass flows. When these parameters are provided, they override the
        base object's calculated values.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator.
        """
        # Handle total_*_mass_flow parameters specially since they're properties
        # on the base object and are used for curve reshaping
        optional_params = ['total_liquid_mass_flow', 'total_gas_mass_flow']
        temp_stochastic_dict = {}
        
        # Temporarily remove these parameters before calling parent _set_stochastic
        for param in optional_params:
            if param in self._StochasticModel__stochastic_dict:
                temp_stochastic_dict[param] = self._StochasticModel__stochastic_dict.pop(param)
        
        # Call parent _set_stochastic for standard parameters
        super()._set_stochastic(seed)
        
        # Manually handle the total_*_mass_flow parameters
        # These are properties on the base object, so we handle them specially
        
        # Define the custom getattr function that retrieves property values
        def get_property_value(obj, prop_name):
            return getattr(obj, prop_name)
        
        for param, value in temp_stochastic_dict.items():
            if value is not None:
                # Determine the type of input and validate appropriately
                if isinstance(value, tuple):
                    if len(value) == 2:
                        setattr(self, param, self._validate_tuple_length_two(
                            param, value, getattr=get_property_value
                        ))
                    elif len(value) == 3:
                        setattr(self, param, self._validate_tuple_length_three(
                            param, value, getattr=get_property_value
                        ))
                    else:
                        setattr(self, param, self._validate_tuple(
                            param, value, getattr=get_property_value
                        ))
                elif isinstance(value, list):
                    setattr(self, param, self._validate_list(
                        param, value, getattr=get_property_value
                    ))
                elif isinstance(value, (int, float)):
                    setattr(self, param, self._validate_scalar(
                        param, value, getattr=get_property_value
                    ))
                else:
                    # Assume it's a custom sampler or similar
                    setattr(self, param, [value])
            else:
                # If None, set as a list with None to indicate no randomization
                setattr(self, param, [None])
            # Restore to stochastic dict
            self._StochasticModel__stochastic_dict[param] = value
        
        # Reset nested stochastic geometry
        if self._is_stochastic_geometry(self.geometry):
            self.geometry._set_stochastic(seed)

    def _resolve_geometry(self):
        """Return a valid geometry object, whether stochastic or static.

        Returns
        -------
        CylindricalTank
            A CylindricalTank object, either created from a stochastic geometry
            or returned directly if it's already a static geometry.
        """
        if self._is_stochastic_geometry(self.geometry):
            return self.geometry.create_object()
        return self.geometry

    def dict_generator(self):
        """Special generator for the mass flow rate based tank class that
        yields a dictionary with the randomly generated input arguments. This
        overrides the base dict_generator to exclude nested stochastic geometry
        objects from being stored in last_rnd_dict, preventing JSON
        serialization issues.

        Additionally, if geometry is stochastic, this method populates
        last_rnd_dict with the geometry's randomized parameters for complete
        traceability.

        Yields
        ------
        dict
            Dictionary with the randomly generated input arguments.
        """
        generated_dict = next(super().dict_generator())

        # Store geometry's randomized parameters if it's stochastic
        if self._is_stochastic_geometry(self.geometry):
            # Store reference to geometry's last_rnd_dict for traceability
            generated_dict["geometry"] = None  # Avoid JSON serialization issues
            generated_dict["geometry_parameters"] = (
                self.geometry.last_rnd_dict.copy()
                if hasattr(self.geometry, "last_rnd_dict")
                else None
            )
        else:
            generated_dict["geometry"] = None

        self.last_rnd_dict = generated_dict
        yield generated_dict

    def create_object(self, max_attempts=100):
        """Creates and returns a MassFlowRateBasedTank object from the randomly
        generated input arguments. If validation fails (overfilled/underfilled),
        retries with new random parameters.

        Parameters
        ----------
        max_attempts : int, optional
            Maximum number of attempts to generate a valid tank configuration.
            Default is 100.

        Returns
        -------
        MassFlowRateBasedTank
            MassFlowRateBasedTank object with the randomly generated input
            arguments.

        Raises
        ------
        ValueError
            If unable to generate a valid tank configuration after max_attempts
            tries. This indicates the parameter ranges are likely incompatible
            (e.g., consistently overfilled or underfilled tanks).
        """
        for attempt in range(max_attempts):
            generated_dict = next(self.dict_generator())

            # Resolve geometry (create if stochastic, use directly if static)
            geometry = self._resolve_geometry()

            # If geometry was stochastic, store its randomized parameters
            if self._is_stochastic_geometry(self.geometry):
                self.last_rnd_dict["geometry_parameters"] = (
                    self.geometry.last_rnd_dict.copy()
                )

            # Get the base mass flow rate functions from the original tank
            liquid_flow_in = self.obj.liquid_mass_flow_rate_in
            liquid_flow_out = self.obj.liquid_mass_flow_rate_out
            gas_flow_in = self.obj.gas_mass_flow_rate_in
            gas_flow_out = self.obj.gas_mass_flow_rate_out

            # Determine if we need to reshape the mass flow curves
            reshape_needed = (
                generated_dict.get("total_liquid_mass_flow") is not None
                or generated_dict.get("total_gas_mass_flow") is not None
            )

            if reshape_needed:
                new_flux_time = (
                    generated_dict["flux_start_time"],
                    generated_dict["flux_stop_time"],
                )

                # Reshape liquid mass flow curves if total_liquid_mass_flow is provided
                if generated_dict.get("total_liquid_mass_flow") is not None:
                    try:
                        liquid_flow_in, liquid_flow_out = self._reshape_mass_flow_curve(
                            liquid_flow_in,
                            liquid_flow_out,
                            new_flux_time,
                            generated_dict["total_liquid_mass_flow"],
                        )
                    except ValueError as e:
                        if attempt == max_attempts - 1:
                            raise ValueError(
                                f"Failed to reshape liquid mass flow curves: {str(e)}"
                            ) from e
                        continue

                # Reshape gas mass flow curves if total_gas_mass_flow is provided
                if generated_dict.get("total_gas_mass_flow") is not None:
                    try:
                        gas_flow_in, gas_flow_out = self._reshape_mass_flow_curve(
                            gas_flow_in,
                            gas_flow_out,
                            new_flux_time,
                            generated_dict["total_gas_mass_flow"],
                        )
                    except ValueError as e:
                        if attempt == max_attempts - 1:
                            raise ValueError(
                                f"Failed to reshape gas mass flow curves: {str(e)}"
                            ) from e
                        continue

            try:
                tank = MassFlowRateBasedTank(
                    name=generated_dict["name"],
                    geometry=geometry,
                    flux_time=(
                        generated_dict["flux_start_time"],
                        generated_dict["flux_stop_time"],
                    ),
                    liquid=generated_dict["liquid"],
                    gas=generated_dict["gas"],
                    initial_liquid_mass=generated_dict["initial_liquid_mass"],
                    initial_gas_mass=generated_dict["initial_gas_mass"],
                    liquid_mass_flow_rate_in=liquid_flow_in,
                    liquid_mass_flow_rate_out=liquid_flow_out,
                    gas_mass_flow_rate_in=gas_flow_in,
                    gas_mass_flow_rate_out=gas_flow_out,
                )
                # If successful, return the valid tank
                return tank
            except ValueError as e:
                # Log the failed attempt (only on last attempt)
                if attempt == max_attempts - 1:
                    error_msg = (
                        f"\n{'='*70}\n"
                        f"Validation Error in StochasticMassFlowRateBasedTank\n"
                        f"{'='*70}\n"
                        f"Failed to generate valid tank after {max_attempts} attempts.\n"
                        f"Last error: {str(e)}\n\n"
                        f"Last attempted parameters:\n"
                        f"  - Flux time: ({generated_dict['flux_start_time']:.3f}, "
                        f"{generated_dict['flux_stop_time']:.3f}) s\n"
                        f"  - Initial liquid mass: {generated_dict['initial_liquid_mass']:.3f} kg\n"
                        f"  - Initial gas mass: {generated_dict['initial_gas_mass']:.3f} kg\n"
                    )
                    if generated_dict.get("total_liquid_mass_flow") is not None:
                        error_msg += f"  - Total liquid mass flow: {generated_dict['total_liquid_mass_flow']:.3f} kg\n"
                    if generated_dict.get("total_gas_mass_flow") is not None:
                        error_msg += f"  - Total gas mass flow: {generated_dict['total_gas_mass_flow']:.3f} kg\n"
                    if self._is_stochastic_geometry(self.geometry):
                        error_msg += (
                            f"\nLast attempted geometry parameters:\n"
                            f"  - Radius: {self.last_rnd_dict['geometry_parameters'].get('radius', 'N/A')} m\n"
                            f"  - Height: {self.last_rnd_dict['geometry_parameters'].get('height', 'N/A')} m\n"
                        )
                    error_msg += (
                        f"\nSuggestion: Adjust the stochastic parameter ranges to avoid "
                        f"consistently invalid tank configurations (overfilled/underfilled).\n"
                        f"Consider reducing initial mass ranges or increasing geometry size.\n"
                        f"{'='*70}"
                    )
                    raise ValueError(error_msg) from e
                # Otherwise, continue to next attempt
                continue

        # This should never be reached due to the exception above
        raise ValueError("Unexpected error in tank generation")

