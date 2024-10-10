import numpy as np
import matplotlib.pyplot as plt


class ALKCostsSingliticoModel():

    def __init__(
        self,
        elec_location: int,
    ):
        self.IF = 0.27 # instatllation fraction [% RC_elec]
        self.RP_elec = 10 # reference power [MW]

        # Values for OpEx taken from [1], Table B.3, PEMEL.
        self.RP_SR = 5 # reference power [MW]
        self.RU_SR = 0.45 # reference cost share [%], for a reference power, RP_SR, of 5MW
        self.P_stack_max_bar = 4 # average max size [MW]
        self.SF_SR_0 = 0.12 # average scale factor


        self.OS = elec_location 
    def run(
        self,
        P_elec: float,
        RC_elec: float,
    ) -> tuple:
        """
        Computes the CapEx and OpEx costs for a single electrolyzer.

        Args:
            P_elec (float): Nominal capacity of the electrolyzer [GW].
            RC_elec (float): Reference cost of the electrolyzer [MUSD/GW] for a 10 MW electrolyzer plant installed.

        Returns:
            tuple: CapEx and OpEx costs for a single electrolyzer.
        """
        capex = self.calc_capex(P_elec, RC_elec)
        
        opex = self.calc_opex(P_elec, capex)

        return capex, opex
    def calc_capex(
        self,
        P_elec: float,
        RC_elec: float,
    ) -> float:
        """
        CapEx for a single electrolyzer, given the electrolyzer capacity and reference cost.
        Equation from [1], Table B.1, CapEx_EL. For in-turbine electrolyzers,
        it is assumed that the maximum electrolyzer size is equal to the turbine rated capacity.

        NOTE: If the single electrolyzer capacity exceeds 100MW, the CapEx becomes fixed at the cost of a
        100MW system, due to decreasing economies of scale (based on assumption from [1]).
        As such, if you use the output to calculate a cost per unit of electrolyzer, you will need to divide
        the cost by 100MW and not the user-specified size of the electrolyzer for sizes above 100 MW.

        Args:
            P_elec (float): Nominal capacity of the electrolyzer [GW].
            RC_elec (float): Reference cost of the electrolyzer [MUSD/GW].

        Returns:
            float: CapEx for electrolyzer [MUSD].
        """
        # Choose the scale factor based on electrolyzer size, [1], Table B.2.
        if P_elec < 10 / 10**3:
            self.SF_elec = -0.24 # scale factor, -0.21 for <10MW, -0.14 for >10MW
        else:
            self.SF_elec = -0.13 # scale factor, -0.21 for <10MW, -0.14 for >10MW
        
        # If electrolyzer capacity is >100MW, fix unit cost to 100MW electrolyzer as economies of scale
        # stop at sizes above this, according to assumption in [1].
        if P_elec > 100 / 10**3:
            P_elec_cost_per_unit_calc = 0.1
        else:
            P_elec_cost_per_unit_calc = P_elec

        # Return the cost of a single electrolyzer of the specified capacity in millions of USD (or the supplied currency).
        # MUSD = GW   * MUSD/GW *           -             *      GW   * MW/GW /      MW       **      -
        cost = P_elec_cost_per_unit_calc * RC_elec * (1 + self.IF * self.OS) *  ((P_elec_cost_per_unit_calc * 10**3 / self.RP_elec) ** self.SF_elec)
        cost_per_unit = cost / P_elec_cost_per_unit_calc

        return cost_per_unit * P_elec
    def calc_opex(
        self,
        P_elec: float,
        capex_elec: float,
        RC_elec: float = None,
        OH: float = None,
    ) -> float:
        """
        OpEx for a single electrolyzer, given the electrolyzer capacity and reference cost.
        Equations from [1], Table B.1, OpEx_elec_eq and OpEx_elec_neq.
        The returned OpEx cost include equipment and non-equipment costs, but excludes the stack replacement cost.

        NOTE: If the single electrolyzer capacity exceeds 100MW, the OpEx becomes fixed at the cost of a
        100MW system, due to decreasing economies of scale (based on assumption from [1]).
        As such, if you use the output to calculate a cost per unit of electrolyzer, you will need to divide
        the cost by 100MW and not the user-specified size of the electrolyzer for sizes above 100 MW.

        NOTE: Code for the stack replacement cost is included below, but does not currently match results
        from [1]. DO NOT USE in the current form.

        Args:
            P_elec (float): Nominal capacity of the electrolyzer [GW].
            capex_elec (float): CapEx for electrolyzer [MUSD].
            RC_elec (float, optional): Reference cost of the electrolyzer [MUSD/GW]. Defaults to None. Not currently used.
            OH (float, optional): Operating hours [h]. Defaults to None. Not currently used.

        Returns:
            float: OpEx for electrolyzer [MUSD].
        """
        # If electrolyzer capacity is >100MW, fix unit cost to 100MW electrolyzer as economies of scale
        # stop at sizes above this, according to assumption in [1].
        if P_elec > 100 / 10**3:
            P_elec = 0.1

        # Including material cost for planned and unplanned maintenance, labor cost in central Europe, which
        # all depend on a system scale. Excluding the cost of electricity and the stack replacement,
        # calculated separately. Scaled maximum to P_elec_bar = 1 GW.
        # MUSD*MW         MUSD    *              -                *    -   *    GW   * MW/GW
        opex_elec_eq = capex_elec * (1 - self.IF * (1 + self.OS)) * 0.0344 * (P_elec * 10**3) ** -0.155

        # Covers the other operational expenditure related to the facility level. This includes site
        # management, land rent and taxes, administrative fees (insurance, legal fees...), and site maintenance.
        # MUSD                    MUSD     
        opex_elec_neq = 0.04 * capex_elec * self.IF * (1 + self.OS)
        return opex_elec_eq + opex_elec_neq