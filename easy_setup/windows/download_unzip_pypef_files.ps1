Write-Host "Downloading and extracting PyPEF (test) files..."
(New-Object Net.WebClient).DownloadFile("https://github.com/niklases/PyPEF/archive/refs/heads/main.zip", "main.zip")
Expand-Archive .\main.zip .