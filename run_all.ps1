
$folderpath = ".\times"

if (-not (Test-Path -Path $folderpath)) {
    New-Item -ItemType Directory -Path $folderpath | Out-Null
}

$inputs = @(3,4,5,6,7,8,9,10)

# Loop over each .exe file
Get-ChildItem -Path . -Filter "*.exe" | ForEach-Object {
    $exefile = $_.Name

    # Match the category from filename
    if ($exefile -match "multiplication-([a-zA-Z_]+)\.exe") {
        $category = $matches[1]
        Write-Host "Executing: $exefile"

        foreach ($inp in $inputs) {
            $resultfile = "$folderpath\results_${category}_${inp}.txt"
            Write-Host "Saving to: $resultfile"

            for ($i = 1; $i -le 10; $i++) {
                Write-Host "Run #$i of $exefile with input $inp"
                $output = $inp | & $_.FullName
                Write-Host $output
                $output | Out-File -Append -Encoding utf8 $resultfile
            }
        }
    }
}
