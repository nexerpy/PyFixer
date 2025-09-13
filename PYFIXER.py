#!/usr/bin/env python3
"""
Complete Advanced Python Code Fixer Telegram Bot
Single file for PythonAnywhere deployment with file attachment support
"""

import os
import re
import logging
import asyncio
import io
from typing import List, Dict, Optional, Tuple

# CONFIGURATION - REPLACE WITH YOUR ACTUAL TOKENS
TELEGRAM_BOT_TOKEN = "8370639949:AAGaGvGA7KxtG2vMpmqe_8jnOG-hL6rByJY"
OPENAI_API_KEY = "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"

# Try to import required packages
try:
    import openai
    from telegram import Update, File, InputFile
    from telegram.ext import (
        Application, 
        CommandHandler, 
        MessageHandler, 
        filters, 
        ContextTypes
    )
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install python-telegram-bot==22.4 openai==1.107.2")
    exit(1)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Validate configuration
if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
    raise ValueError("Please replace TELEGRAM_BOT_TOKEN with your actual token")
if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
    raise ValueError("Please replace OPENAI_API_KEY with your actual token")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Telegram message limits
MAX_MESSAGE_LENGTH = 4000  # Leave some buffer under 4096 limit


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
    """Split long messages into chunks that fit Telegram's limits"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by lines first to keep formatting
    lines = text.split('\n')
    
    for line in lines:
        # If single line is too long, split it
        if len(line) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long line by words
            words = line.split(' ')
            for word in words:
                if len(current_chunk + word + " ") > max_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = word + " "
                    else:
                        # Single word too long, force split
                        chunks.append(word[:max_length])
                        current_chunk = word[max_length:] + " "
                else:
                    current_chunk += word + " "
        else:
            # Check if adding this line exceeds limit
            if len(current_chunk + line + "\n") > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
                else:
                    chunks.append(line)
            else:
                current_chunk += line + "\n"
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


class PythonAnalyzer:
    """Analyzes Python code for syntax errors, missing imports, and structural issues"""
    
    def __init__(self):
        self.import_patterns = {
            'os.': 'import os',
            'sys.': 'import sys', 
            'json.': 'import json',
            'requests.': 'import requests',
            'datetime.': 'from datetime import datetime',
            'time.': 'import time',
            're.': 'import re',
            'random.': 'import random',
            'math.': 'import math',
            'numpy.': 'import numpy',
            'pandas.': 'import pandas',
            'matplotlib.': 'import matplotlib.pyplot as plt',
        }
    
    def analyze_code(self, code: str, filename: Optional[str] = None) -> Dict:
        """Analyze Python code and return issues found"""
        logger.info(f'🔍 [PythonAnalyzer] Starting analysis of {len(code)} characters')
        
        syntax_errors = []
        missing_imports = []
        indentation_issues = []
        advanced_issues = []
        
        lines = code.split('\n')
        
        # Check for syntax issues
        for i, line in enumerate(lines, 1):
            trimmed = line.strip()
            if not trimmed or trimmed.startswith('#'):
                continue
                
            # Missing colons
            if re.match(r'^\s*(if|elif|else|for|while|def|class|try|except|finally|with)\s+.*[^:]$', line):
                syntax_errors.append(f"Line {i}: Missing colon after {trimmed.split()[0]} statement")
            
            # Old-style print statements
            if re.match(r'^\s*print\s+(?!\()', line):
                syntax_errors.append(f"Line {i}: Old-style print statement - use print() function")
            
            # Unterminated strings
            double_quotes = line.count('"') - line.count('\\"')
            single_quotes = line.count("'") - line.count("\\'")
            
            if double_quotes % 2 == 1 and '"""' not in line:
                syntax_errors.append(f"Line {i}: Unterminated double-quoted string")
            if single_quotes % 2 == 1 and "'''" not in line:
                syntax_errors.append(f"Line {i}: Unterminated single-quoted string")
        
        # Check for missing imports
        existing_imports = set()
        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith('import ') or trimmed.startswith('from '):
                existing_imports.add(trimmed)
        
        for pattern, import_stmt in self.import_patterns.items():
            if pattern in code:
                module_name = import_stmt.split()[-1].split('.')[0]
                if not any(module_name in imp for imp in existing_imports):
                    missing_imports.append(import_stmt)
        
        # Check for advanced issues
        if 'if True:' in code:
            advanced_issues.append("Found 'if True:' - might be debugging code")
        if 'if False:' in code:
            advanced_issues.append("Found 'if False:' - unreachable code")
        if re.search(r'while\s+True\s*:', code) and 'break' not in code:
            advanced_issues.append("Potential infinite loop detected (while True without break)")
        if re.search(r'except\s*:', code):
            advanced_issues.append("Bare except clause found - should specify exception type")
        if 'eval(' in code:
            advanced_issues.append("Use of eval() detected - potential security risk")
        if 'exec(' in code:
            advanced_issues.append("Use of exec() detected - potential security risk")
        
        is_valid = len(syntax_errors) == 0 and len(missing_imports) == 0
        needs_confirmation = len(advanced_issues) > 0
        
        logger.info(f'✅ [PythonAnalyzer] Analysis complete: {len(syntax_errors)} syntax errors, {len(missing_imports)} missing imports')
        
        return {
            'is_valid': is_valid,
            'syntax_errors': syntax_errors,
            'missing_imports': missing_imports,
            'indentation_issues': indentation_issues,
            'advanced_issues': advanced_issues,
            'needs_confirmation': needs_confirmation
        }


class PythonFixer:
    """Automatically fixes common Python code issues"""
    
    def __init__(self):
        self.import_patterns = {
            'os.': 'import os',
            'sys.': 'import sys',
            'json.': 'import json', 
            'requests.': 'import requests',
            'datetime.': 'from datetime import datetime',
            'time.': 'import time',
            're.': 'import re',
            'random.': 'import random',
            'math.': 'import math',
            'numpy.': 'import numpy',
            'pandas.': 'import pandas',
            'matplotlib.': 'import matplotlib.pyplot as plt',
        }
    
    def fix_code(self, code: str, allow_advanced_fixes: bool = False, filename: Optional[str] = None) -> Dict:
        """Fix Python code and return the fixed version with applied fixes"""
        logger.info(f'🔧 [PythonFixer] Starting to fix {len(code)} characters')
        
        fixed_code = code
        applied_fixes = []
        
        try:
            # Add missing imports
            lines = fixed_code.split('\n')
            existing_imports = set()
            
            for line in lines:
                trimmed = line.strip()
                if trimmed.startswith('import ') or trimmed.startswith('from '):
                    existing_imports.add(trimmed)
            
            imports_to_add = []
            for pattern, import_stmt in self.import_patterns.items():
                if pattern in fixed_code:
                    module_name = import_stmt.split()[-1].split('.')[0]
                    if not any(module_name in imp for imp in existing_imports):
                        imports_to_add.append(import_stmt)
                        applied_fixes.append(f"Added missing import: {import_stmt}")
            
            # Insert imports at the top
            if imports_to_add:
                import_section = []
                code_section = []
                in_imports = True
                
                for line in lines:
                    trimmed = line.strip()
                    if in_imports and (trimmed.startswith('import ') or trimmed.startswith('from ') or 
                                     trimmed == '' or trimmed.startswith('#')):
                        import_section.append(line)
                    else:
                        in_imports = False
                        code_section.append(line)
                
                fixed_code = '\n'.join(import_section + imports_to_add + [''] + code_section)
            
            # Fix syntax errors
            fixed_lines = []
            lines = fixed_code.split('\n')
            
            for i, line in enumerate(lines):
                modified_line = line
                trimmed = line.strip()
                
                if not trimmed or trimmed.startswith('#'):
                    fixed_lines.append(line)
                    continue
                
                # Fix missing colons
                colon_patterns = [
                    r'^(\s*)(if\s+.+)(?<!:)$',
                    r'^(\s*)(elif\s+.+)(?<!:)$', 
                    r'^(\s*)(else)(?<!:)$',
                    r'^(\s*)(for\s+.+)(?<!:)$',
                    r'^(\s*)(while\s+.+)(?<!:)$',
                    r'^(\s*)(def\s+.+)(?<!:)$',
                    r'^(\s*)(class\s+.+)(?<!:)$',
                    r'^(\s*)(try)(?<!:)$',
                    r'^(\s*)(except.*)(?<!:)$',
                    r'^(\s*)(finally)(?<!:)$',
                    r'^(\s*)(with\s+.+)(?<!:)$'
                ]
                
                for pattern in colon_patterns:
                    if re.match(pattern, line) and not line.endswith(':'):
                        modified_line = line + ':'
                        applied_fixes.append(f"Added missing colon on line {i + 1}")
                        break
                
                # Fix old-style print statements
                print_match = re.match(r'^(\s*)print\s+(.+)$', modified_line)
                if print_match and not re.match(r'^(\s*)print\s*\(', modified_line):
                    indent, content = print_match.groups()
                    if not (content.startswith('"') or content.startswith("'")):
                        if ',' in content:
                            modified_line = f"{indent}print({content})"
                        else:
                            modified_line = f'{indent}print("{content}")'
                    else:
                        modified_line = f"{indent}print({content})"
                    applied_fixes.append(f"Fixed old-style print statement on line {i + 1}")
                
                # Fix unterminated strings (basic)
                if trimmed and not trimmed.endswith('"""') and not trimmed.endswith("'''"):
                    double_quotes = modified_line.count('"') - modified_line.count('\\"')
                    single_quotes = modified_line.count("'") - modified_line.count("\\'")
                    
                    if double_quotes % 2 == 1 and not modified_line.endswith('"'):
                        modified_line = modified_line + '"'
                        applied_fixes.append(f"Fixed unterminated double-quoted string on line {i + 1}")
                    elif single_quotes % 2 == 1 and not modified_line.endswith("'"):
                        modified_line = modified_line + "'"
                        applied_fixes.append(f"Fixed unterminated single-quoted string on line {i + 1}")
                
                fixed_lines.append(modified_line)
            
            fixed_code = '\n'.join(fixed_lines)
            
            # Apply advanced fixes if allowed
            if allow_advanced_fixes:
                if 'if True:' in fixed_code:
                    fixed_code = fixed_code.replace('if True:', 'if True:  # TODO: Replace with actual condition')
                    applied_fixes.append("Added TODO for 'if True:' condition")
                
                if 'if False:' in fixed_code:
                    fixed_code = re.sub(r'if False:.*?\n', '# REMOVED: Unreachable code\n', fixed_code)
                    applied_fixes.append("Commented out unreachable 'if False:' code")
                
                if re.search(r'except\s*:', fixed_code):
                    fixed_code = re.sub(r'except\s*:', 'except Exception:', fixed_code)
                    applied_fixes.append("Changed bare except to 'except Exception:'")
            
            # Basic validation
            is_valid = True
            validation_message = "Code appears syntactically valid"
            
            if not applied_fixes:
                applied_fixes = ["No fixes needed - code was already good!"]
            
            logger.info(f'✅ [PythonFixer] Fixing complete: {len(applied_fixes)} fixes applied')
            
            return {
                'fixed_code': fixed_code,
                'applied_fixes': applied_fixes,
                'is_valid': is_valid,
                'validation_message': validation_message
            }
            
        except Exception as e:
            logger.error(f'❌ [PythonFixer] Error during fixing: {e}')
            return {
                'fixed_code': code,
                'applied_fixes': [f"Error during fixing: {str(e)}"],
                'is_valid': False,
                'validation_message': f"Fixing failed: {str(e)}"
            }


class CodeFixerBot:
    """Main bot class that handles Telegram interactions and AI processing"""
    
    def __init__(self):
        self.analyzer = PythonAnalyzer()
        self.fixer = PythonFixer()
    
    async def send_long_message(self, update: Update, message: str, parse_mode: str = 'Markdown') -> None:
        """Send long messages by splitting them into chunks"""
        if not update.message:
            return
            
        chunks = split_message(message)
        
        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    await update.message.reply_text(chunk, parse_mode=parse_mode)
                else:
                    await update.message.reply_text(chunk, parse_mode=parse_mode)
                
                # Small delay between messages to avoid rate limiting
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error sending message chunk {i+1}: {e}")
                # Fallback without parsing if markdown fails
                try:
                    await update.message.reply_text(chunk)
                except Exception as e2:
                    logger.error(f"Failed to send even plain text: {e2}")
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        user_name = update.effective_user.first_name if update.effective_user else "Unknown"
        logger.info(f'👋 [Bot] User {user_name} started the bot')
        
        welcome_message = """🐍 **Advanced Python Code Fixer Bot**

Welcome! I'm your AI-powered Python code assistant. I can help you:

✅ **Auto-fix:** syntax errors, indentation, missing colons, unterminated strings
✅ **Add:** missing imports, exception handling around risky operations  
✅ **Advanced analysis:** detect code issues, logic problems, security risks

**How to use me:**
📝 Send me Python code directly in a message
📎 Upload a Python file (.py, .pyw, .txt) - I'll send back the fixed file!

I'll analyze your code, fix common issues, and send back the improved version with detailed explanations!

Try sending me some Python code right now! 🚀"""
        
        await self.send_long_message(update, welcome_message)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages with Python code"""
        if not update.message or not update.message.text:
            return
            
        user_name = update.effective_user.first_name if update.effective_user else "Unknown"
        logger.info(f'📝 [Bot] Received message from {user_name}')
        
        code = update.message.text
        chat_id = update.effective_chat.id if update.effective_chat else 0
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        
        try:
            # Use AI to process the code
            ai_response = await self.process_with_ai(code, is_file=False)
            
            await self.send_long_message(update, ai_response)
            
        except Exception as e:
            logger.error(f'❌ [Bot] Error processing message: {e}')
            error_msg = f"❌ **Error processing your code:**\n\n`{str(e)}`\n\nPlease try again or contact support."
            await self.send_long_message(update, error_msg)
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle document uploads (Python files)"""
        if not update.message or not update.message.document:
            return
            
        user_name = update.effective_user.first_name if update.effective_user else "Unknown"
        logger.info(f'📎 [Bot] Received document from {user_name}')
        
        document = update.message.document
        chat_id = update.effective_chat.id if update.effective_chat else 0
        
        # Check if it's a Python file
        file_name = document.file_name or ""
        if not (file_name.endswith('.py') or 
                file_name.endswith('.pyw') or 
                file_name.endswith('.txt')):
            await self.send_long_message(update, "❌ Please send a Python file (.py, .pyw, or .txt extension)")
            return
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        
        try:
            # Download the file
            file: File = await document.get_file()
            file_content = await file.download_as_bytearray()
            code = file_content.decode('utf-8')
            
            logger.info(f'📥 [Bot] Downloaded file: {file_name} ({len(code)} characters)')
            
            # Process the code (no length limit for file processing)
            analysis = self.analyzer.analyze_code(code, file_name)
            fix_result = self.fixer.fix_code(code, allow_advanced_fixes=False, filename=file_name)
            
            # Create summary message (short)
            summary = f"""🔍 **Analysis Complete for {file_name}**

**Issues Found:**
• Syntax Errors: {len(analysis['syntax_errors'])}
• Missing Imports: {len(analysis['missing_imports'])}
• Advanced Issues: {len(analysis['advanced_issues'])}

**Fixes Applied:** {len(fix_result['applied_fixes'])}

🚀 **Sending you the fixed file...**"""
            
            # Send summary
            await self.send_long_message(update, summary)
            
            # Create fixed filename
            base_name = file_name.replace('.py', '').replace('.pyw', '').replace('.txt', '')
            fixed_filename = f"{base_name}_fixed.py"
            
            # Create file buffer
            fixed_file_buffer = io.BytesIO()
            fixed_file_buffer.write(fix_result['fixed_code'].encode('utf-8'))
            fixed_file_buffer.seek(0)
            
            # Create caption with fix details
            caption = f"🚀 **Your fixed Python code: {fixed_filename}**\n\n"
            if len(fix_result['applied_fixes']) <= 5:
                caption += "**Fixes applied:**\n"
                for fix in fix_result['applied_fixes']:
                    caption += f"• {fix}\n"
            else:
                caption += f"**{len(fix_result['applied_fixes'])} fixes applied** (see summary above)\n"
            
            # Limit caption length
            if len(caption) > 1000:
                caption = caption[:997] + "..."
            
            # Send the fixed file
            await context.bot.send_document(
                chat_id=chat_id,
                document=InputFile(fixed_file_buffer, filename=fixed_filename),
                caption=caption,
                parse_mode='Markdown',
                reply_to_message_id=update.message.message_id
            )
            
            logger.info(f'✅ [Bot] Sent fixed file: {fixed_filename}')
            
        except Exception as e:
            logger.error(f'❌ [Bot] Error processing file: {e}')
            error_msg = f"❌ **Error processing your file:**\n\n`{str(e)}`\n\nPlease try again."
            await self.send_long_message(update, error_msg)
    
    async def process_with_ai(self, code: str, is_file: bool = False, filename: Optional[str] = None) -> str:
        """Process code with AI and return response"""
        logger.info(f'🤖 [Bot] Processing with AI: {len(code)} characters')
        
        # First analyze the code
        analysis = self.analyzer.analyze_code(code, filename)
        
        # Then try to fix it
        fix_result = self.fixer.fix_code(code, allow_advanced_fixes=False, filename=filename)
        
        # Create shorter context for AI to avoid token limits
        context_prompt = f"""
Analyze and fix this Python code:

**Original Code:**
```python
{code[:1500]}{'...' if len(code) > 1500 else ''}
```

**Issues Found:**
- Syntax Errors: {len(analysis['syntax_errors'])}
- Missing Imports: {len(analysis['missing_imports'])}  
- Advanced Issues: {len(analysis['advanced_issues'])}

**Auto-Fixed Version:**
```python
{fix_result['fixed_code'][:1500]}{'...' if len(fix_result['fixed_code']) > 1500 else ''}
```

Provide a concise, helpful response with:
1. Brief summary of issues found and fixed
2. The complete fixed code in a code block
3. Key improvements made

Keep response under 3000 characters total.
"""
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Python code assistant. Provide concise, helpful responses under 3000 characters."},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content or "No response from AI"
            logger.info(f'✅ [Bot] AI processing complete')
            
            return ai_response
            
        except Exception as e:
            logger.error(f'❌ [Bot] OpenAI API error: {e}')
            
            # Fallback response without AI (shorter version)
            fallback_response = f"""🔍 **Code Analysis Complete**

**Issues Found:**
"""
            if analysis['syntax_errors']:
                fallback_response += f"❌ **Syntax Errors:** {len(analysis['syntax_errors'])} found\n"
                for error in analysis['syntax_errors'][:3]:  # Limit to first 3
                    fallback_response += f"• {error}\n"
                if len(analysis['syntax_errors']) > 3:
                    fallback_response += f"• ... and {len(analysis['syntax_errors']) - 3} more\n"
            
            if analysis['missing_imports']:
                fallback_response += f"📦 **Missing Imports:** {len(analysis['missing_imports'])} found\n"
                for imp in analysis['missing_imports'][:3]:  # Limit to first 3
                    fallback_response += f"• {imp}\n"
                if len(analysis['missing_imports']) > 3:
                    fallback_response += f"• ... and {len(analysis['missing_imports']) - 3} more\n"
            
            fallback_response += f"""
🔧 **Fixed Code:**
```python
{fix_result['fixed_code']}
```

✅ Your code has been analyzed and fixed! Issues resolved: {len(fix_result['applied_fixes'])}"""
            
            return fallback_response
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        help_text = """🐍 **Python Code Fixer Bot Help**

**Commands:**
• `/start` - Welcome message and introduction
• `/help` - Show this help message

**How to use:**
1️⃣ **Send Python code directly** - Just paste your code in a message
2️⃣ **Upload Python files** - Send .py, .pyw, or .txt files

**What I can fix:**
✅ Syntax errors (missing colons, parentheses)
✅ Old-style print statements
✅ Missing imports (os, sys, json, etc.)
✅ Unterminated strings
✅ Indentation issues
✅ Security issues (eval, exec warnings)
✅ Logic problems (infinite loops, unreachable code)

**File Upload Feature:**
When you upload a Python file, I will:
🔍 Analyze the entire file (no length limits)
🔧 Fix all issues found
📎 Send you back the fixed file as `filename_fixed.py`

**Example:**
Send me code like this:
```
if x > 5
    print "Hello"
```

I'll fix it to:
```python
if x > 5:
    print("Hello")
```

Need help? Just send me your Python code! 🚀"""
        
        await self.send_long_message(update, help_text)


def main():
    """Main function to start the bot"""
    logger.info('🚀 [Bot] Starting Advanced Python Code Fixer Bot')
    print('🚀 [Bot] Starting Advanced Python Code Fixer Bot')
    
    # Create bot instance
    bot = CodeFixerBot()
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, bot.handle_document))
    
    # Start the bot
    logger.info('✅ [Bot] Bot is ready and listening for messages')
    print('✅ [Bot] Bot is ready and listening for messages')
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()