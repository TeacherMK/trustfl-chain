$ErrorActionPreference = "Stop"
$runs = @(
  "python src/main.py --dataset fashion_mnist --rounds 5 --attack label_flip --defense fedavg --non_iid true",
  "python src/main.py --dataset fashion_mnist --rounds 5 --attack label_flip --defense trustfl_chain --non_iid true",
  "python src/main.py --dataset fashion_mnist --rounds 5 --attack sign_flip --defense median --non_iid true",
  "python src/main.py --dataset fashion_mnist --rounds 5 --attack gaussian_noise --defense trimmed_mean --non_iid true",
  "python src/main.py --dataset synthetic --rounds 3 --attack model_scaling --defense trustfl_chain --non_iid true --no_download"
)
foreach ($run in $runs) {
  Write-Host "Running: $run"
  Invoke-Expression $run
}
