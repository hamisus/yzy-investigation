# Discord Manager

A comprehensive Discord server management tool that provides functionality for:
- Server backups with message history
- Channel management
- Message summarization
- User management

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

The message summarization feature is currently under development. It will allow you to:
- Generate daily recaps of channel activity
- Create summaries of specific time periods
- Filter summaries by keywords or topics

## Development

The project is structured as follows:
- `src/discord_manager.py`: Main Discord manager class
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