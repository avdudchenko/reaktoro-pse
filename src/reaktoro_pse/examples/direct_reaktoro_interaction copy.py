# import base classes
from pyomo.environ import ConcreteModel, Var, units as pyunits

from reaktoro_pse.core.reaktoro_state import (
    ReaktoroState,
)

from reaktoro_pse.core.reaktoro_inputs import (
    ReaktoroInputSpec,
)

import reaktoro as rkt

if __name__ == "__main__":
    m = ConcreteModel()
    m.temp = Var(initialize=278.15, units=pyunits.K)
    m.pressure = Var(initialize=1e5, units=pyunits.Pa)

    comp = {
        "H2O": 21.986114415912887,
        "H+": 2.016224760825584e-07,
        "OH-": 6.746347740901594e-09,
        "MgOH+": 1.470712869766662e-08,
        "HCO3-": 0.0036699342066493037,
        "HSO4-": 1.4596347214236603e-07,
        # "modifier_Ca(OH)2": 0.0035961774139978545,
        "CO3-2": 5.372374059242281e-06,
        "CO2": 0.0019273908162560393,
        "MgCO3": 3.4723912715458604e-06,
        "SO4-2": 0.10835135626543013,
        "Na+": 0.0038245557517450597,
        "Mg+2": 0.08252280662824542,
        "Cl-": 0.046116645496774446,
        "K+": 0.0008854783875549223,
        "Ca+2": 0.04837154248289573,
    }

    m.composition = Var(
        list(comp.keys()),
        initialize=1,
        units=pyunits.mol / pyunits.s,
    )
    for key, val in comp.items():
        m.composition[key] = val
    # setup reaktoro state
    rkt_state = ReaktoroState()
    rkt_state.set_database(database="pitzer.dat")
    rkt_state.set_input_options("aqueous_phase", convert_to_rkt_species=False)
    rkt_state.register_system_inputs(temperature=m.temp, pressure=m.pressure)
    rkt_state.register_aqueous_inputs(composition=m.composition)
    rkt_state.set_aqueous_phase_activity_model(activity_model="ActivityModelPitzer")
    rkt_state.build_state()

    # equlibrate it
    rkt_state.equilibrate_state()
    print(rkt_state.state)

    # print(rkt.AqueousProps(rkt_state.state.props()))

    rkt_input = ReaktoroInputSpec(rkt_state)
    rkt_input.configure_specs(dissolve_species_in_rkt=True, exact_speciation=True)
    rkt_input.register_charge_neutrality(assert_neutrality=False, ion="Na")
    # register chemical
    m.slaked_lime = Var(initialize=0.003596177413997854, units=pyunits.mol / pyunits.s)
    # rkt_input.register_chemistry_modifier("Ca(OH)2", m.slaked_lime)
    rkt_input.build_input_specs()

    solver = rkt.EquilibriumSolver(rkt_input.equilibrium_specs)
    conditions = rkt.EquilibriumConditions(rkt_input.equilibrium_specs)
    conditions.temperature(m.temp.value)
    conditions.pressure(m.pressure.value)

    # any input key has "input" as starting string
    # conditions.set("inputmodifier_Ca(OH)2", 0.0)  # m.slaked_lime.value)

    for key, val in comp.items():
        # if "Na" not in key:
        conditions.set(f"input{key}", val / 10)
    # conditions.charge(0)

    # define solver options and logging
    solver_options = rkt.EquilibriumOptions()

    solver_options.epsilon = 1e-32
    solver_options.optima.maxiters = 600
    solver_options.optima.output.active = True
    solver.setOptions(solver_options)

    result = solver.solve(rkt_state.state, conditions)
    print(rkt_state.state)

    print(rkt.AqueousProps(rkt_state.state.props()))
