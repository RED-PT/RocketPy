from rocketpy import MassFlowRateBasedTank, Fluid, CylindricalTank

from rocketpy.stochastic.stochastic_model import StochasticModel

from rocketpy import Tail
from rocketpy.stochastic import StochasticTail

class StochasticCylindricalTank(StochasticModel):
    
    def __init__(
        self,
        cylindrical_tank,
        radius=None,
        height=None
    ):
        """Initializes the Stochastic Cylindrical Tank class.

        See Also
        --------
        :ref:`stochastic_model` and :class:`CylindricalTank <rocketpy.tanks.CylindricalTank>`

        Parameters
        ----------
        cylindrical_tank : CylindricalTank
            CylindricalTank object to be used for validation.
        radius : int, float, tuple, list, optional
            Radius of the tank in meters.
        height : int, float, tuple, list, optional
            Height of the tank in meters.
        spherical_caps : bool, optional
            Whether the tank has spherical caps.
        """
        self.spherical_caps = cylindrical_tank.has_caps
        super().__init__(
            cylindrical_tank,
            radius=radius,
            height=height
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
            spherical_caps=self.spherical_caps
        )

class StochasticMassFlowRateBasedTank(StochasticModel):

    def __init__(
        self,
        mass_flow_rate_based_tank,
        geometry,
        flux_start_time=None,
        flux_stop_time=None,
        liquid=None,
        gas=None,
        initial_liquid_mass=None,
        initial_gas_mass=None
    ):
        
        self.geometry = geometry
        
        super().__init__(
            mass_flow_rate_based_tank,
            flux_start_time=flux_start_time,
            flux_stop_time=flux_stop_time,
            liquid=None,
            gas=None,
            initial_liquid_mass=initial_liquid_mass,
            initial_gas_mass=initial_gas_mass,
            gas_mass_flow_rate_in=None,
            gas_mass_flow_rate_out=None,
            liquid_mass_flow_rate_in=None,
            liquid_mass_flow_rate_out=None,
            name=None
        )

    def _resolve_geometry(self):
        """Return a valid geometry object, whether stochastic or static."""
        if isinstance(self.geometry, StochasticModel):
            return self.geometry.create_object()
        else:
            return self.geometry

    def create_object(self):
        """Creates and returns a MassFlowRateBasedTank object from the randomly generated
        input arguments.

        Returns
        -------
        MassFlowRateBasedTank
            MassFlowRateBasedTank object with the randomly generated input arguments.
        """
        generated_dict = next(self.dict_generator())

        geometry = self._resolve_geometry()

        return MassFlowRateBasedTank(
            name=generated_dict["name"],
            geometry=geometry,
            flux_time=(generated_dict['flux_start_time'], generated_dict['flux_stop_time']),
            liquid=generated_dict['liquid'],
            gas=generated_dict['gas'],
            initial_liquid_mass=generated_dict['initial_liquid_mass'],
            initial_gas_mass=generated_dict['initial_gas_mass'],
            liquid_mass_flow_rate_in=generated_dict['liquid_mass_flow_rate_in'],
            liquid_mass_flow_rate_out=generated_dict['liquid_mass_flow_rate_out'],
            gas_mass_flow_rate_in=generated_dict['gas_mass_flow_rate_in'],
            gas_mass_flow_rate_out=generated_dict['gas_mass_flow_rate_out']
        )


        

