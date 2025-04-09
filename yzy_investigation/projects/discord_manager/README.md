# Discord Manager

A comprehensive Discord server management tool that provides functionality for:
- Server backups with message history
- Channel management
- Message summarization
- User management
- Summary publishing to Discord channels

## Setup

1. Create a Discord bot and get your bot token:
   - Go to the [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application
   - Go to the "Bot" section and create a bot
   - Copy the bot token

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your bot token and server ID:
   ```
   DISCORD_BOT_TOKEN=your_bot_token_here
   DISCORD_SERVER_ID=your_server_id_here
   DISCORD_CHANNEL_ID=your_channel_id_here
   
   # Optional test environment
   DISCORD_SERVER_ID_TEST=your_test_server_id_here
   DISCORD_CHANNEL_ID_TEST=your_test_channel_id_here
   ```

5. Invite the bot to your server:
   - Go to the OAuth2 section in the Discord Developer Portal
   - Select the following scopes:
     - `bot`
     - `applications.commands`
   - Select the following bot permissions:
     - `Read Messages/View Channels`
     - `Send Messages`
     - `Read Message History`
     - `Attach Files`
   - Use the generated URL to invite the bot to your server

## Usage

### Backup Messages

To backup messages from your server:

```bash
# Backup messages from the last 24 hours
python src/cli.py backup

# Backup all messages
python src/cli.py backup --all

# Backup messages from a specific date range
python src/cli.py backup --start-date 2024-01-01 --end-date 2024-01-31
```

### Message Summarization

The message summarization feature allows you to generate summaries of Discord channel activity:

```bash
# Generate daily recap for default channels
python src/cli.py daily-recap

# Generate recap for a specific time period
python src/cli.py daily-recap --start-time "2024-04-01 00:00:00" --end-time "2024-04-02 00:00:00"
```

### Publishing Summaries to Discord

You can publish summary bullet points from generated summaries to a Discord channel:

```bash
# Publish a summary to the Discord channel specified in .env
python src/cli.py publish-summary data/discord/summaries/game-building_summary.md

# Publish to test environment
python src/cli.py publish-summary data/discord/summaries/game-building_summary.md --test-mode

# Customize publishing behavior
python src/cli.py publish-summary data/discord/summaries/game-building_summary.md --delay 2.0 --include-overview
```

Options:
- `--test-mode`: Use test server and channel IDs specified in the .env file
- `--include-overview`: Include the overview paragraph as the first message (default: true)
- `--delay`: Delay between messages in seconds to avoid rate limiting (default: 1.0)

## Development

The project is structured as follows:
- `src/discord_manager.py`: Main Discord manager class
- `src/message_summarizer.py`: Handles summarization of Discord messages
- `src/summary_publisher.py`: Publishes summary bullet points to Discord channels
- `src/daily_recap.py`: Generates daily recaps of Discord channel activity
- `src/cli.py`: Command-line interface
- `requirements.txt`: Project dependencies
- `.env`: Configuration file (not tracked in git)

## Contributing

Feel free to contribute to the project by:
- Adding new features
- Improving existing functionality
- Fixing bugs
- Writing documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details. 