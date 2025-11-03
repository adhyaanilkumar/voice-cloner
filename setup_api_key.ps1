# Setup ElevenLabs API Key
# Run this script to set your API key for the current PowerShell session

$apiKey = "sk_ba45f5ccfd343a60ac150b37944f9f94565970206d83edec"

# Set environment variable for current session
$env:ELEVENLABS_API_KEY = $apiKey

Write-Host "ElevenLabs API key has been set for this PowerShell session!" -ForegroundColor Green
Write-Host "API Key: $($apiKey.Substring(0,10))..." -ForegroundColor Gray
Write-Host ""
Write-Host "To make this permanent, add the following to your PowerShell profile:" -ForegroundColor Yellow
Write-Host "  `$env:ELEVENLABS_API_KEY = '$apiKey'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or create a .env file in the project root with:" -ForegroundColor Yellow
Write-Host "  ELEVENLABS_API_KEY=$apiKey" -ForegroundColor Cyan
Write-Host ""

