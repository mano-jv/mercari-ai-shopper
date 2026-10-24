# Configure this repository to use the mano-jv GitHub identity for commits and pushes.
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$includePath = "../.github/gitconfig.mano-jv"
$included = git config --local --get-all include.path

if ($included -notcontains $includePath) {
    git config --local --add include.path $includePath
}

git config --local user.name "mano-jv"
git config --local user.email "mano-jv@users.noreply.github.com"
git config --local github.user "mano-jv"
git config --local credential.https://github.com/mano-jv.username "mano-jv"
git remote set-url origin "https://github.com/mano-jv/mercari-ai-shopper.git"

Write-Host "Configured local git identity for mano-jv."
Write-Host ""
Write-Host "Active repo settings:"
git config --local --get user.name
git config --local --get user.email
git config --local --get github.user
git remote -v
Write-Host ""

$ghStatus = gh auth status --hostname github.com 2>&1 | Out-String
if ($ghStatus -notmatch "account mano-jv") {
    Write-Host "GitHub CLI is not authenticated as mano-jv yet."
    Write-Host "Run: gh auth login --hostname github.com --git-protocol https --web"
    Write-Host "Then switch to mano-jv if prompted, or add that account with gh auth login again."
}
