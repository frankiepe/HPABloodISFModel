using PackageCompiler

create_sysimage(
    [:HPADDEModels, :DelayDiffEq, :DifferentialEquations, :JSON];
    sysimage_path="julia_hpa_sysimage.so",
    project="."
)
