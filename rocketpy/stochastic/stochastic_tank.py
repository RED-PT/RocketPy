"""Defines the StochasticCylindricalTank and StochasticMassFlowRateBasedTank classes."""

from rocketpy.motors.tank import CylindricalTank, MassFlowRateBasedTank

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
    radius : int, float, tuple, list
        Radius of the tank in meters.
    height : int, float, tuple, list
        Height of the tank in meters.
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
            Radius of the tank in meters.
        height : int, float, tuple, list, optional
            Height of the tank in meters.
        """
        self.spherical_caps = cylindrical_tank.has_caps
        super().__init__(
            cylindrical_tank,
            radius=radius,
            height=height,
        )

    def create_object(self):
        """Creates and returns a CylindricalTank object from the randomly generated
        input arguments.

        Returns
        -------
        CylindricalTank
            CylindricalTank object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())

        return CylindricalTank(
            radius=generated_dict["radius"],
            height=generated_dict["height"],
            spherical_caps=self.spherical_caps,
        )


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
            gas_mass_flow_rate_in=None,
            gas_mass_flow_rate_out=None,
            liquid_mass_flow_rate_in=None,
            liquid_mass_flow_rate_out=None,
            name=None,
        )

    def _resolve_geometry(self):
        """Return a valid geometry object, whether stochastic or static.
        
        Returns
        -------
        CylindricalTank
            A CylindricalTank object, either created from a stochastic geometry
            or returned directly if it's already a static geometry.
        """
        if isinstance(self.geometry, StochasticModel):
            return self.geometry.create_object()
        return self.geometry

    def dict_generator(self):
        """Special generator for the mass flow rate based tank class that
        yields a dictionary with the randomly generated input arguments. This
        overrides the base dict_generator to exclude nested stochastic geometry
        objects from being stored in last_rnd_dict, preventing JSON
        serialization issues.

        Yields
        ------
        dict
            Dictionary with the randomly generated input arguments.
        """
        generated_dict = next(super().dict_generator())
        # Replace geometry with None to avoid storing StochasticCylindricalTank
        # objects, which would cause JSON serialization errors in MonteCarlo
        if "geometry" in generated_dict:
            generated_dict["geometry"] = None
        self.last_rnd_dict = generated_dict
        yield generated_dict

    def create_object(self):
        """Creates and returns a MassFlowRateBasedTank object from the randomly
        generated input arguments.

        Returns
        -------
        MassFlowRateBasedTank
            MassFlowRateBasedTank object with the randomly generated input
            arguments.
        """
        generated_dict = next(self.dict_generator())

        geometry = self._resolve_geometry()

        return MassFlowRateBasedTank(
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
            liquid_mass_flow_rate_in=generated_dict["liquid_mass_flow_rate_in"],
            liquid_mass_flow_rate_out=generated_dict["liquid_mass_flow_rate_out"],
            gas_mass_flow_rate_in=generated_dict["gas_mass_flow_rate_in"],
            gas_mass_flow_rate_out=generated_dict["gas_mass_flow_rate_out"],
        )


        

