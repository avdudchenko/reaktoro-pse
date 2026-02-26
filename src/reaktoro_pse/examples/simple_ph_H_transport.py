from reaktoro_pse.reaktoro_block import ReaktoroBlock
from reaktoro_pse.core.util_classes.cyipopt_solver import (
    get_cyipopt_watertap_solver,
)
from pyomo.environ import (
    ConcreteModel,
    Var,
    value,
    Block,
    Constraint,
    log10,
    units as pyunits,
    assert_optimal_termination,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
import idaes.core.util.scaling as iscale

import reaktoro as rkt

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    ActivityCoefficientModel,
    DensityCalculation,
)
from idaes.models.unit_models import (
    Product,
    Feed,
    StateJunction,
)
import watertap.property_models.seawater_prop_pack as props_seawater

import watertap.property_models.NaCl_T_dep_prop_pack as props_nacl
from idaes.core import (
    FlowsheetBlock,
)

import numpy as np
from idaes.core.util.model_statistics import degrees_of_freedom
import csv

from reaktoro_enabled_watertap.utils.report_util import get_lib_path
from reaktoro_enabled_watertap.water_sources.source_water_importer import (
    get_source_water_data,
)

from watertap.core.solvers import get_solver


def main():
    m = build_naive_pH_model_and_solve()
    print(degrees_of_freedom(m))
    solver = get_solver()
    result = solver.solve(m, tee=True)

    for r in [2, 4, 6, 8, 10]:
        m.fs.H_transport.fix(r * 1e-8)
        result = solver.solve(m, tee=False)
        print(
            [
                "H transport (mol/s)",
                r * 1e-8,
                "pH_naive",
                m.fs.delta_block.pH.value,
            ]
        )
        assert_optimal_termination(result)


def build_naive_pH_model(
    blk,
    block_name="pH_block",
):
    blk.add_component(block_name, Block())
    ph_block = blk.find_component(block_name)
    ph_block.element_mol_comps = Var(
        ["H", "O"], initialize=1, units=pyunits.mol / pyunits.s
    )
    ph_block.specie_mol_comps = Var(
        ["H2O", "H_+", "OH_-"], initialize=1, units=pyunits.mol / pyunits.s
    )
    ph_block.specie_mol_conc = Var(
        ["H2O", "H_+", "OH_-"], initialize=1, units=pyunits.mol / pyunits.L
    )
    ph_block.solution_mass = Var(initialize=1, units=pyunits.kg / pyunits.s)
    ph_block.volume = Var(initialize=1, units=pyunits.L / pyunits.s)
    ph_block.density = Var(initialize=1, units=pyunits.kg / pyunits.L)
    ph_block.density.fix()
    ph_block.h2o_dissociation_k1 = Var(initialize=2e-5, units=pyunits.s**-1)
    ph_block.h2o_dissociation_k1.fix()
    ph_block.h_oh_association_k2 = Var(
        initialize=1.5e14, units=pyunits.cm**3 / pyunits.mol / pyunits.s
    )
    ph_block.h_oh_association_k2.fix()
    ph_block.pH = Var(initialize=7, bounds=(0, 14), units=pyunits.dimensionless)

    # ensure elemental balance is maintained,
    # this will allow us to solve for pH based on water dissociation and feed water composition,
    # without needing to speciate the full solution,
    # this is critical for larger databases where speciation can be very expensive.
    @ph_block.Constraint(["H", "O"])
    def eq_element_balance(b, specie):
        if specie == "H":
            return b.element_mol_comps["H"] == (
                2 * b.specie_mol_comps["H2O"]
                + b.specie_mol_comps["H_+"]
                - b.specie_mol_comps["OH_-"]
            )
        elif specie == "O":
            return b.element_mol_comps["O"] == (
                b.specie_mol_comps["H2O"]
                + b.specie_mol_comps["OH_-"]
                - b.specie_mol_comps["H_+"]
            )

    @ph_block.Constraint(["H2O", "H_+", "OH_-"])
    def eq_specie_mol_comps(b, specie):
        return b.specie_mol_comps[specie] == b.specie_mol_conc[specie] * b.volume

    ph_block.eq_solution_mass = Constraint(
        expr=ph_block.solution_mass
        == sum(ph_block.specie_mol_comps[specie] for specie in ["H2O", "H_+", "OH_-"])
        * 18e-3
    )
    ph_block.eq_volume = Constraint(
        expr=ph_block.volume == ph_block.solution_mass / ph_block.density
    )
    # H20 -> H+ + OH- and H+ + OH- -> H2O
    ph_block.h2o_equilibrium = Constraint(
        expr=ph_block.h2o_dissociation_k1 * ph_block.specie_mol_comps["H2O"]
        == ph_block.h_oh_association_k2
        * ph_block.specie_mol_comps["H_+"]
        * ph_block.specie_mol_comps["OH_-"]
    )

    ph_block.eq_pH = Constraint(
        expr=ph_block.pH == -log10(ph_block.specie_mol_conc["H_+"])
    )


def scale_ph_block(ph_block, water_amount):
    iscale.set_scaling_factor(ph_block.specie_mol_comps["H2O"], 1 / water_amount)
    iscale.set_scaling_factor(ph_block.specie_mol_comps["H_+"], 1 / water_amount)
    iscale.set_scaling_factor(ph_block.specie_mol_comps["OH_-"], 1 / water_amount)
    iscale.set_scaling_factor(ph_block.pH, 1)
    iscale.set_scaling_factor(ph_block.h2o_dissociation_k1, 1e5)
    iscale.set_scaling_factor(ph_block.h_oh_association_k2, 1e-14)


def initialize_ph_block(ph_block, water_amount, initial_pH=7):
    ph_block.specie_mol_comps["H2O"].fix(water_amount)
    ph_block.pH.fix(initial_pH)
    calculate_variable_from_constraint(ph_block.specie_mol_conc["H_+"], ph_block.eq_pH)
    solver = get_solver()
    result = solver.solve(ph_block, tee=True)
    ph_block.display()
    assert_optimal_termination(result)


def build_naive_pH_model_and_solve(water_amount=55.5, initial_pH=7):
    m = ConcreteModel()

    m.fs = FlowsheetBlock()
    build_naive_pH_model(m.fs, "initial_block")
    build_naive_pH_model(m.fs, "delta_block")
    initialize_ph_block(m.fs.initial_block, water_amount, initial_pH)
    initialize_ph_block(m.fs.delta_block, water_amount, initial_pH)

    m.fs.delta_block.pH.unfix()
    m.fs.delta_block.specie_mol_comps["H2O"].unfix()
    m.fs.H_transport = Var(initialize=0, units=pyunits.mol / pyunits.s)
    m.fs.H_transport.fix()

    iscale.set_scaling_factor(m.fs.H_transport, 1e10)
    m.fs.OH_transport = Var(initialize=0, units=pyunits.mol / pyunits.s)
    m.fs.OH_transport.fix()

    iscale.set_scaling_factor(m.fs.OH_transport, 1e10)

    @m.fs.Constraint(["H", "O"])
    def eq_H_OH_transport(b, specie):
        if specie == "H":
            return (
                b.delta_block.element_mol_comps["H"]
                == b.initial_block.element_mol_comps["H"]
                + b.H_transport
                + b.OH_transport
            )
        elif specie == "O":
            return (
                b.delta_block.element_mol_comps["O"]
                == b.initial_block.element_mol_comps["O"] + b.OH_transport
            )

    return m


def print_comparison(m):
    header = [
        "Temperature (K)",
        "TDS",
        #  "Reaktoro_osmoticPressure (Pa)",
        # "Reaktoro_vaporPressure (Pa)",
        "Reaktoro pH",
    ]
    for phase, ion in m.fs.multicomp_feed.properties[0].conc_mass_phase_comp:
        header.append(f"{ion} (mg/L)")

    data_row = [
        m.fs.multicomp_feed.properties[0].temperature.value,
        m.fs.feed_tds.value,
        # m.fs.modified_properties["osmoticPressure", "H2O"].value,
        # m.fs.modified_properties["vaporPressure", "H2O(g)"].value,
        m.fs.modified_properties["pH", None].value,
    ]
    # for phase, ion in m.fs.multicomp_feed.properties[0].conc_mass_phase_comp:
    #     data_row.append(
    #         value(
    #             pyunits.convert(
    #                 m.fs.multicomp_feed.properties[0].conc_mass_phase_comp[phase, ion],
    #                 to_units=pyunits.mg / pyunits.L,
    #             )
    #         )
    #     )
    result = {}
    str_print = ""
    for header, val in zip(header, data_row):
        str_print += f"{header}: {float(val)} "
        result[header] = float(val)
    print(str_print)
    return result


def build_model(water_case):

    mcas_props, feed_specs = get_source_water_data(water_case)
    m = ConcreteModel()
    m.fs = FlowsheetBlock()
    m.fs.mcas_props = MCASParameterBlock(**mcas_props)
    m.fs.multicomp_feed = Feed(property_package=m.fs.mcas_props)
    m.fs.multicomp_feed.properties[0].flow_mol_phase_comp[...]
    m.fs.multicomp_feed.properties[0].total_dissolved_solids[...]
    m.fs.multicomp_feed.properties[0].flow_vol_phase[...]

    m.fs.feed_pH = Var(
        initialize=feed_specs["pH"], bounds=(4, 12), units=pyunits.dimensionless
    )
    m.fs.feed_pH.fix(7)
    m.fs.H_transport = Var(initialize=5e-8, units=pyunits.mol / pyunits.s)
    m.fs.H_transport.fix()
    iscale.set_scaling_factor(m.fs.H_transport, 1e8)
    m.fs.OH_transport = Var(initialize=0, units=pyunits.mol / pyunits.s)
    iscale.set_scaling_factor(m.fs.OH_transport, 1e8)
    m.fs.OH_transport.fix()
    # add global TDS constraint
    set_feed_composition(m, feed_specs)

    # make sure TDS is equal across all property packages
    total_solids = []
    for i in m.fs.multicomp_feed.properties[0].flow_mass_phase_comp:
        if "H2O" not in i[1]:
            total_solids.append(
                m.fs.multicomp_feed.properties[0].flow_mass_phase_comp[i]
            )

    # fix the mass flows of each ion and unfix water mass flow, it wil be adjusted to match TDS,
    # similarly we unfix Cl in multicomp to allow charge balance
    for idx in m.fs.multicomp_feed.properties[0].conc_mass_phase_comp:
        m.fs.multicomp_feed.properties[0].conc_mass_phase_comp[idx].unfix()
        m.fs.multicomp_feed.properties[0].flow_mass_phase_comp[idx].fix()

    print(degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0
    result = solve_model(m)
    assert_optimal_termination(result)
    return m


def set_feed_composition(m, feed_specs):
    temperature = feed_specs["temperature"]
    m.fs.multicomp_feed.properties[0].temperature.fix(temperature * pyunits.K)
    m.fs.multicomp_feed.properties[0].pressure.fix(101325 * pyunits.Pa)

    for ion in feed_specs["ion_concentrations"]:
        m.fs.multicomp_feed.properties[0].conc_mass_phase_comp["Liq", ion].fix(
            feed_specs["ion_concentrations"][ion]
        )
        m.fs.multicomp_feed.properties[0].flow_mol_phase_comp["Liq", ion].unfix()
    m.fs.multicomp_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(
        1 * pyunits.kg / pyunits.s
    )
    m.fs.multicomp_feed.properties[0].flow_mol_phase_comp["Liq", "H2O"].unfix()

    assert degrees_of_freedom(m.fs.multicomp_feed) == 0
    result = solve_model(m.fs.multicomp_feed)
    scale_model(m)

    assert degrees_of_freedom(m.fs.multicomp_feed) == 0
    result = solve_model(m.fs.multicomp_feed)
    assert_optimal_termination(result)
    add_standard_properties(m)
    m.fs.eq_modified_properties.initialize()
    assert degrees_of_freedom(m) == 0
    result = solve_model(m)
    assert_optimal_termination(result)


def add_standard_properties(
    m,
    database="pitzer.dat",
    activity_model="ActivityModelPitzer",
):
    m.fs.modified_properties = Var(
        [
            ("pH", None),
        ],
        initialize=1,
    )
    prop_dict = {
        ("density", None): m.fs.multicomp_feed.properties[0].dens_mass_phase["Liq"],
    }
    for key in m.fs.modified_properties:
        prop_dict[key] = m.fs.modified_properties[key]
    m.fs.multicomp_feed.properties[0].dens_mass_phase["Liq"].unfix()
    # this will charge nutralize the solution and remove water as vapor, allowing assessment of how pressure (osmotic/or vapor changes as function of tempearuter and recovery)
    m.fs.eq_modified_properties = ReaktoroBlock(
        aqueous_phase={
            "composition": m.fs.multicomp_feed.properties[0].flow_mol_phase_comp,
            "convert_to_rkt_species": True,
            "activity_model": activity_model,
        },
        database_file=database,
        system_state={
            "temperature": m.fs.multicomp_feed.properties[0].temperature,
            "pressure": m.fs.multicomp_feed.properties[0].pressure,
            "pH": m.fs.feed_pH,
        },
        outputs=prop_dict,
        chemistry_modifier={"OH": m.fs.H_transport},  # , "OH": m.fs.OH_transport},
        # assert_charge_neutrality=False,
        # we can use default converter as its defined for default database (Phreeqc and pitzer)
        # we are modifying state and must speciate inputs before adding acid to find final prop state.
        build_speciation_block=True,  # direct calculations here
    )
    # m.fs.multicomp_feed.properties[0].conc_mass_phase_comp["Liq", "Cl_-"].unfix()
    # m.fs.initial_neutral_ion = (
    #     m.fs.multicomp_feed.properties[0].conc_mass_phase_comp["Liq", "Cl_-"].value
    # )

    m.fs.multicomp_feed.properties[0].eq_dens_mass_phase["Liq"].deactivate()


def scale_model(m):

    tds_scale = 0
    for idx in m.fs.multicomp_feed.properties[0].flow_mol_phase_comp:
        scale = 1 / m.fs.multicomp_feed.properties[0].flow_mol_phase_comp[idx].value
        if "H2O" not in idx[1]:
            tds_scale += (
                m.fs.multicomp_feed.properties[0].flow_mass_phase_comp[idx].value
            )
        m.fs.mcas_props.set_default_scaling("flow_mol_phase_comp", scale, index=idx)

    iscale.calculate_scaling_factors(m)


def solve_model(m, tee=True, **kwargs):
    cy_solver = get_cyipopt_watertap_solver()  # get_solver(solver="cyipopt-watertap")
    result = cy_solver.solve(m, tee=True)
    return result


if __name__ == "__main__":
    main()
