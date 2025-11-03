# Vercel Deployment - Summary of Changes

## Files Created/Modified

### New Files Created:
1. **`vercel.json`** - Vercel configuration file
   - Routes all requests to the FastAPI handler
   - Configures Python 3.9 runtime
   - Includes web_dashboard files in deployment

2. **`api/index.py`** - Vercel serverless function handler
   - Imports and exports the FastAPI app
   - Sets up proper path resolution for imports

3. **`.vercelignore`** - Files to exclude from deployment
   - Excludes cache files, logs, databases, generated files
   - Reduces deployment size

4. **`requirements-vercel.txt`** - Minimal dependencies for Vercel
   - Excludes heavy ML libraries (torch, tensorflow) that exceed size limits
   - Includes only essential dependencies for ElevenLabs API functionality

5. **`VERCEL_DEPLOYMENT.md`** - Comprehensive deployment guide
   - Step-by-step deployment instructions
   - Troubleshooting tips
   - Environment variable setup

### Modified Files:
1. **`src/api/main.py`** - Updated for serverless compatibility
   - Added serverless environment detection
   - Updated logging to skip file logging in serverless
   - Updated file paths to use `/tmp` in serverless environments
   - Made database initialization optional in serverless

2. **`src/database/database.py`** - Updated for serverless
   - Uses `/tmp` for SQLite in serverless environments
   - Note: SQLite is not ideal for serverless - consider using external DB

## Key Changes Made

### Serverless Environment Detection
- Detects Vercel environment via `VERCEL` environment variable
- Adapts file paths, logging, and database handling accordingly

### File Path Handling
- Local: Uses `generated_audio/` directory
- Serverless: Uses `/tmp/generated_audio/` directory
- Files in `/tmp` are ephemeral and cleared between invocations

### Logging
- Local: Uses both StreamHandler and FileHandler
- Serverless: Uses only StreamHandler (no file logging)

### Database
- Local: Uses `voice_cloner.db` in project root
- Serverless: Uses `/tmp/voice_cloner.db` (SQLite not ideal for serverless)
- Database initialization is optional in serverless (warns but doesn't fail)

### Static Files
- Served through FastAPI routes, not Vercel's static file handler
- Web dashboard files are included in deployment via `vercel.json`

## Deployment Steps

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Set Environment Variables:**
   ```bash
   vercel env add ELEVENLABS_API_KEY
   ```

4. **Deploy:**
   ```bash
   vercel --prod
   ```

## Important Notes

### Size Limitations
- Vercel free tier has function size limits
- Heavy ML libraries (torch, tensorflow) are excluded from `requirements-vercel.txt`
- Core functionality (ElevenLabs API) doesn't require these libraries

### Database Considerations
- SQLite is not ideal for serverless (stateless functions)
- Consider using:
  - Vercel Postgres
  - Supabase
  - PlanetScale
  - Or make database operations optional (already handled)

### State Management
- Serverless functions are stateless
- Don't rely on in-memory state between requests
- Use external storage for persistence

### Cold Starts
- First request may be slower (cold start)
- Subsequent requests are faster (warm function)

## Testing Locally with Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Run local development server
vercel dev
```

This will simulate the Vercel environment locally.

## Next Steps

1. Review `VERCEL_DEPLOYMENT.md` for detailed instructions
2. Set up environment variables in Vercel dashboard
3. Deploy using `vercel --prod`
4. Monitor function logs in Vercel dashboard
5. Consider setting up a proper database for production use

