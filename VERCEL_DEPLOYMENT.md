# Vercel Deployment Guide

This guide explains how to deploy the Voice Cloner application on Vercel.

## Prerequisites

1. A Vercel account ([sign up here](https://vercel.com/signup))
2. Vercel CLI installed (`npm i -g vercel`)
3. ElevenLabs API key

## Deployment Steps

### 1. Install Vercel CLI (if not already installed)

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Set Environment Variables

Set your ElevenLabs API key in Vercel:

**Via Vercel Dashboard:**
1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add `ELEVENLABS_API_KEY` with your API key

**Via CLI:**
```bash
vercel env add ELEVENLABS_API_KEY
# Enter your API key when prompted
```

### 4. Deploy to Vercel

```bash
# Deploy to production
vercel --prod

# Or deploy to preview
vercel
```

### 5. Configure Build Settings

Vercel will automatically detect the Python runtime. Make sure:
- **Framework Preset**: Other
- **Build Command**: (leave empty)
- **Output Directory**: (leave empty)
- **Install Command**: `pip install -r requirements-vercel.txt`

### 6. Update requirements.txt (Optional)

For Vercel deployment, you can either:
- Use `requirements-vercel.txt` (minimal dependencies)
- Or use the full `requirements.txt` (may exceed size limits)

**Note**: Heavy ML libraries like `torch` and `tensorflow` may not work on Vercel's free tier due to size limitations. Since this application uses ElevenLabs API, these are not required for core functionality.

## Project Structure for Vercel

```
.
├── api/
│   └── index.py          # Vercel serverless handler
├── vercel.json           # Vercel configuration
├── .vercelignore         # Files to ignore in deployment
├── requirements-vercel.txt # Minimal dependencies for Vercel
└── src/                  # Application source code
```

## Configuration Files

### vercel.json
Routes all requests to the FastAPI application handler.

### api/index.py
Serverless function handler that imports and exposes the FastAPI app.

### .vercelignore
Excludes unnecessary files from deployment (logs, databases, cache, etc.).

## Important Notes

### Serverless Limitations

1. **File System**: 
   - Use `/tmp` directory for temporary files (limited to 512MB)
   - Files in `/tmp` are ephemeral and cleared between invocations

2. **Database**:
   - SQLite is not ideal for serverless (stateless functions)
   - Consider using:
     - Vercel Postgres
     - Supabase
     - PlanetScale
     - Or make database operations optional

3. **State**:
   - Serverless functions are stateless
   - Don't rely on in-memory state between requests
   - Use external storage for persistence

4. **Cold Starts**:
   - First request may be slower (cold start)
   - Subsequent requests are faster (warm function)

### Environment Variables

Required environment variables:
- `ELEVENLABS_API_KEY`: Your ElevenLabs API key

Optional environment variables:
- `DATABASE_URL`: For database connection (if using external DB)
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000, not used in serverless)

### Function Size Limits

- **Free Tier**: 50MB uncompressed
- **Pro Tier**: 250MB uncompressed
- **Enterprise**: Custom limits

Large dependencies like `torch` and `tensorflow` may exceed these limits.

## Troubleshooting

### Issue: "Function exceeded maximum size"

**Solution**: Use `requirements-vercel.txt` instead of full `requirements.txt`

### Issue: "Module not found"

**Solution**: Ensure all dependencies are in `requirements-vercel.txt`

### Issue: "Database connection failed"

**Solution**: 
- Use an external database (PostgreSQL, MySQL)
- Or make database operations optional (already handled in code)

### Issue: "Static files not loading"

**Solution**: Static files are served through FastAPI routes, not Vercel's static file handler

## Post-Deployment

After deployment, your application will be available at:
- Production: `https://your-project.vercel.app`
- Preview: `https://your-project-{hash}.vercel.app`

## Monitoring

- Check Vercel dashboard for function logs
- Monitor API usage and function execution times
- Set up alerts for errors

## Support

For issues specific to Vercel deployment:
- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)

