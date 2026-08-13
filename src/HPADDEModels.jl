module HPADDEModels

include("Models.jl")
include("Parameters.jl")

using .Models
export BaseHPAModel, crh

using .Parameters
export day_len

end
