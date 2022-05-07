

abstract type NetworkParameter end

struct GeneratorParameter <: Parameter
    id::String
end

struct BusParameter <: Parameter
    id::String
end

const vma = BusParameter("vm")
const lmp = BusParameter("lam_kcl_r")
const gap = GeneratorParameter("pg")


# const voltage_magnitude = "vm"
# const active_power = "pg"
# const reactive_power = "pmin"
# const acive_power_min = "pmax"
# const active_power_max = "pmax"
# const reactive_power_min = "qmin"
# const reactive_power_max = "qmax"

