# Configuration Directory

This directory contains configuration templates and examples for the Scotch-RAG system.

## Files

- **`config_env_example.txt`** - Example environment configuration file template

## Usage

Copy the example configuration file to create your own:

```bash
cp config/config_env_example.txt .env
```

Then edit the `.env` file with your actual API keys and configuration values.

## Environment Variables

The system requires the following environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `LLAMA_CLOUD_API_KEY` - Your LlamaCloud API key (optional, for LlamaParse)
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: True)

## Notes

- Never commit your actual `.env` file to version control
- The `.env` file should be added to `.gitignore`
- Use `config/environment.local` for local development overrides
