"""
Discord RAG Bot
==============
Discord bot version of the RAG system using the same foundation.
"""

import os
import asyncio
import discord
from discord import app_commands
from discord.ext import commands
from typing import Optional
import traceback
from pathlib import Path

from rag_system_v1 import get_rag_system
from file_monitor import get_file_monitor

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Global instances
rag_system = None
file_monitor = None

class DiscordRAGBot:
    def __init__(self):
        self.rag_system = get_rag_system()
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)
        self.file_monitor = get_file_monitor()
        
        # Start file monitoring
        self.file_monitor.start_monitoring()
        print("✅ Discord RAG Bot initialized with file monitoring")
    
    async def setup_bot_commands(self):
        """Setup Discord slash commands"""
        
        @bot.tree.command(name="ask", description="Ask a question about your data")
        @app_commands.describe(
            question="Your question about the uploaded data"
        )
        async def ask_command(interaction: discord.Interaction, question: str):
            await interaction.response.defer(thinking=True)
            
            try:
                # Process the question
                result = self.rag_system.query(question)
                
                if result["status"] == "success":
                    # Create embed for the answer
                    embed = discord.Embed(
                        title="🤖 RAG Answer",
                        description=result["answer"][:4000],  # Discord limit
                        color=discord.Color.green()
                    )
                    
                    # Add quality metrics (without warnings)
                    if result.get("quality_metrics"):
                        metrics = result["quality_metrics"]
                        embed.add_field(
                            name="📊 Performance",
                            value=f"⏱️ {metrics['query_time']}s | 📝 {metrics['total_tokens']} tokens | 📚 {metrics['source_count']} sources",
                            inline=False
                        )
                    
                    # Add sources
                    if result["sources"]:
                        sources_text = ""
                        for i, source in enumerate(result["sources"][:3], 1):
                            filename = source['metadata'].get('filename', 'Unknown')
                            content_preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                            sources_text += f"**{i}.** {filename}\n{content_preview}\n\n"
                        
                        if sources_text:
                            embed.add_field(
                                name="📚 Sources",
                                value=sources_text[:1024],
                                inline=False
                            )
                    
                    await interaction.followup.send(embed=embed)
                else:
                    error_embed = discord.Embed(
                        title="❌ Error",
                        description=f"Query failed: {result['error']}",
                        color=discord.Color.red()
                    )
                    await interaction.followup.send(embed=error_embed)
                    
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Error",
                    description=f"An error occurred: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
        
        @bot.tree.command(name="upload", description="Upload a file to the RAG system")
        @app_commands.describe(
            file="Excel or PDF file to upload"
        )
        async def upload_command(interaction: discord.Interaction, file: discord.Attachment):
            await interaction.response.defer(thinking=True)
            
            try:
                # Check file type
                allowed_extensions = ['.xlsx', '.xls', '.pdf']
                file_extension = Path(file.filename).suffix.lower()
                
                if file_extension not in allowed_extensions:
                    error_embed = discord.Embed(
                        title="❌ Invalid File Type",
                        description=f"Please upload an Excel (.xlsx, .xls) or PDF file. Got: {file_extension}",
                        color=discord.Color.red()
                    )
                    await interaction.followup.send(embed=error_embed)
                    return
                
                # Download file to data directory
                file_path = self.data_dir / file.filename
                await file.save(file_path)
                
                # Process the file
                result = self.rag_system.ingest_file(str(file_path))
                
                if result["status"] == "success":
                    success_embed = discord.Embed(
                        title="✅ File Uploaded Successfully",
                        description=f"**{result['filename']}** has been processed and added to the RAG system.",
                        color=discord.Color.green()
                    )
                    success_embed.add_field(
                        name="📊 Processing Results",
                        value=f"• Chunks created: {result['chunks_created']}\n• Total characters: {result['total_characters']:,}\n• Pages processed: {result['pages_processed']}",
                        inline=False
                    )
                    await interaction.followup.send(embed=success_embed)
                else:
                    error_embed = discord.Embed(
                        title="❌ Upload Failed",
                        description=f"Failed to process {file.filename}: {result['error']}",
                        color=discord.Color.red()
                    )
                    await interaction.followup.send(embed=error_embed)
                    
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Error",
                    description=f"An error occurred during upload: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
        
        @bot.tree.command(name="stats", description="View system statistics")
        async def stats_command(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            
            try:
                stats = self.rag_system.get_stats()
                
                if stats["status"] == "active":
                    embed = discord.Embed(
                        title="📊 System Statistics",
                        color=discord.Color.blue()
                    )
                    embed.add_field(name="Status", value="🟢 Active", inline=True)
                    embed.add_field(name="Total Vectors", value=str(stats["total_vectors"]), inline=True)
                    embed.add_field(name="Index Size", value=f"{stats['index_size_mb']} MB", inline=True)
                else:
                    embed = discord.Embed(
                        title="📊 System Statistics",
                        description="🔴 No data available. Upload some files first!",
                        color=discord.Color.yellow()
                    )
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Error",
                    description=f"Failed to get statistics: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
        
        @bot.tree.command(name="quality", description="View quality metrics and performance data")
        async def quality_command(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            
            try:
                quality_metrics = self.rag_system.get_quality_metrics()
                
                if quality_metrics.get("status") == "no_data":
                    embed = discord.Embed(
                        title="📊 Quality Metrics",
                        description="No quality data available yet. Ask some questions first!",
                        color=discord.Color.yellow()
                    )
                else:
                    embed = discord.Embed(
                        title="📊 Quality Metrics",
                        color=discord.Color.blue()
                    )
                    embed.add_field(
                        name="Performance",
                        value=f"• Avg Query Time: {quality_metrics['avg_query_time']}s\n• Total Queries: {quality_metrics['total_queries']}\n• Cache Hit Rate: {quality_metrics['cache_hit_rate']:.1%}",
                        inline=True
                    )
                    embed.add_field(
                        name="Quality",
                        value=f"• Avg Confidence: {quality_metrics['avg_confidence']:.3f}\n• Avg Relevance: {quality_metrics['avg_relevance']:.3f}",
                        inline=True
                    )
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Error",
                    description=f"Failed to get quality metrics: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
        
        @bot.tree.command(name="list", description="List uploaded files")
        async def list_command(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True)
            
            try:
                stats = self.rag_system.get_stats()
                
                if stats["status"] == "no_index":
                    embed = discord.Embed(
                        title="📋 Uploaded Files",
                        description="No files uploaded yet",
                        color=discord.Color.yellow()
                    )
                else:
                    # Get file information from the index
                    vectorstore = self.rag_system.load_index()
                    if not vectorstore:
                        embed = discord.Embed(
                            title="📋 Uploaded Files",
                            description="No files found",
                            color=discord.Color.yellow()
                        )
                    else:
                        # Extract unique filenames from metadata
                        filenames = set()
                        for doc in vectorstore.docstore._dict.values():
                            if hasattr(doc, 'metadata') and doc.metadata:
                                filename = doc.metadata.get('filename', 'Unknown')
                                filenames.add(filename)
                        
                        if filenames:
                            files_text = "\n".join([f"• {filename}" for filename in sorted(filenames)])
                            embed = discord.Embed(
                                title="📋 Uploaded Files",
                                description=files_text,
                                color=discord.Color.green()
                            )
                        else:
                            embed = discord.Embed(
                                title="📋 Uploaded Files",
                                description="No files found",
                                color=discord.Color.yellow()
                            )
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Error",
                    description=f"Failed to list files: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
        
        @bot.tree.command(name="reset", description="Reset the RAG system (clear all data)")
        async def reset_command(interaction: discord.Interaction):
            # Check if user has admin permissions
            if not interaction.user.guild_permissions.administrator:
                error_embed = discord.Embed(
                    title="❌ Permission Denied",
                    description="You need administrator permissions to reset the system.",
                    color=discord.Color.red()
                )
                await interaction.response.send_message(embed=error_embed)
                return
            
            await interaction.response.defer(thinking=True)
            
            try:
                self.rag_system.reset()
                success_embed = discord.Embed(
                    title="✅ System Reset",
                    description="All data has been cleared successfully.",
                    color=discord.Color.green()
                )
                await interaction.followup.send(embed=success_embed)
                
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Error",
                    description=f"Failed to reset system: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
        
        @bot.tree.command(name="help", description="Show help information")
        async def help_command(interaction: discord.Interaction):
            help_embed = discord.Embed(
                title="🤖 Discord RAG Bot Help",
                description="This bot uses the same RAG system as the terminal version with Discord integration.",
                color=discord.Color.blue()
            )
            
            help_embed.add_field(
                name="📤 Upload Commands",
                value="• `/upload` - Upload Excel or PDF files\n• Supported: .xlsx, .xls, .pdf",
                inline=False
            )
            
            help_embed.add_field(
                name="❓ Query Commands",
                value="• `/ask <question>` - Ask questions about your data\n• `/ask_web <question>` - Ask with web search (if available)",
                inline=False
            )
            
            help_embed.add_field(
                name="📊 System Commands",
                value="• `/stats` - View system statistics\n• `/quality` - View quality metrics\n• `/list` - List uploaded files\n• `/reset` - Clear all data (admin only)",
                inline=False
            )
            
            help_embed.add_field(
                name="💡 Tips",
                value="• Upload files first before asking questions\n• Use specific questions for better answers\n• Check stats to see your data status\n• Files in ./data/ are auto-processed",
                inline=False
            )
            
            await interaction.response.send_message(embed=help_embed)

# Bot events
@bot.event
async def on_ready():
    print(f"✅ {bot.user} is ready and online!")
    print(f"🔗 Bot is in {len(bot.guilds)} guilds")
    
    # Setup commands
    rag_bot = DiscordRAGBot()
    await rag_bot.setup_bot_commands()
    
    # Sync commands
    try:
        synced = await bot.tree.sync()
        print(f"✅ Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")

@bot.event
async def on_guild_join(guild):
    print(f"🎉 Joined guild: {guild.name} (id: {guild.id})")

@bot.event
async def on_guild_remove(guild):
    print(f"👋 Left guild: {guild.name} (id: {guild.id})")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    
    error_embed = discord.Embed(
        title="❌ Error",
        description=f"An error occurred: {str(error)}",
        color=discord.Color.red()
    )
    await ctx.send(embed=error_embed)

def main():
    """Main function to run the Discord bot"""
    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "RAG_DISCORD_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them before running the bot.")
        return
    
    # Check for optional variables
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("⚠️ LLAMA_CLOUD_API_KEY not set. Excel processing may be limited.")
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️ TAVILY_API_KEY not set. Web search will be disabled.")
    
    print("🔗 Using OpenAI embeddings (paid, high quality)")
    print("🤖 Starting Discord RAG Bot...")
    
    # Run the bot
    bot.run(os.getenv("RAG_DISCORD_TOKEN"))

if __name__ == "__main__":
    main()
