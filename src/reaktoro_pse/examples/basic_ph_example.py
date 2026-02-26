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
from idaes.core import (
    FlowsheetBlock,
)
import idaes.core.util.scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.core.solvers import get_solver


from pyomo.util.calc_var_value import calculate_variable_from_constraint


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


if __name__ == "__main__":
    main()
