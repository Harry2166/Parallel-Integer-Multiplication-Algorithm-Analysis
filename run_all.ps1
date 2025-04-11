
$folderpath = ".\times"

if (-not (Test-Path -Path $folderpath)) {
    New-Item -ItemType Directory -Path $folderpath | Out-Null
}

$inputs = @(3,4,5,6,7,8,9,10)

Get-ChildItem -Path . -Filter "*.exe" | ForEach-Object {
    $exefile = $_.Name

    if ($exefile -match "multiplication-([a-zA-Z_]+)\.exe") {
        $category = $matches[1]
        Write-Host "Executing: $exefile"

        foreach ($inp in $inputs) {
            $txtfile = "$folderpath\results_${category}_${inp}.csv"

            if (Test-Path $txtfile) { Remove-Item $txtfile }

            Add-Content -Path $txtfile -Value "Run,X1Y1,X2Y2,X3Y3,X4Y4,X5Y5,X6Y6,X7Y7,X8Y8,X9Y9,X10Y10,X11Y11,X12Y12,X13Y13,X14Y14,X15Y15,X16Y16,X17Y17,X18Y18,X19Y19,X20Y20,X21Y21,X22Y22,X23Y23,X24Y24,X25Y25"

            for ($i = 1; $i -le 10; $i++) {
                Write-Host "Run #$i of $exefile with input $inp"
                $output = $inp | & $_.FullName
                Write-Host $output

                $line = "$i,$output"
                Add-Content -Path $txtfile -Value $line
            }

            Write-Host "CSV saved to: $txtfile"
        }
    }
}
python gather_all_results.py 
python plotting_all_results.py
