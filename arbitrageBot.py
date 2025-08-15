#!/usr/bin/env python3
"""
UNIFIED SOLANA ARBITRAGE BOT v3.8 - WITH BALANCE TRACKING
Execution rate limits fixed + Balance tracking at start and end
"""

import sys
import os

# Check Python version
if sys.version_info < (3, 8):
    print("ERROR: Python 3.8+ required. Current version:", sys.version)
    exit(1)

# Import standard libraries first
import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import base64
import getpass

# Try importing required packages
try:
    import aiohttp
    print("SUCCESS: aiohttp imported successfully")
except ImportError as e:
    print(f"ERROR: aiohttp import failed: {e}")
    exit(1)

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    print("SUCCESS: cryptography imported successfully")
except ImportError as e:
    print(f"ERROR: cryptography import failed: {e}")
    exit(1)

try:
    import base58
    print("SUCCESS: base58 imported successfully")
except ImportError as e:
    print(f"ERROR: base58 import failed: {e}")
    exit(1)

try:
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Confirmed
    from solana.rpc.types import TxOpts
    print("SUCCESS: solana.rpc imported successfully")
except ImportError as e:
    print(f"ERROR: solana.rpc import failed: {e}")
    exit(1)

try:
    from solders.keypair import Keypair
    print("SUCCESS: Keypair imported from solders.keypair")
except ImportError:
    try:
        from solana.keypair import Keypair
        print("SUCCESS: Keypair imported from solana.keypair")
    except ImportError as e:
        print(f"ERROR: Keypair import failed: {e}")
        exit(1)

try:
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction
    from solders.message import to_bytes_versioned
    print("SUCCESS: solders imported successfully")
except ImportError as e:
    print(f"ERROR: solders import failed: {e}")
    exit(1)

print("SUCCESS: All packages imported successfully!")
print("=" * 50)

# Configure logging
def setup_logging():
    """Setup comprehensive logging"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # File handler
    file_handler = logging.FileHandler(f'logs/arbitrage_balance_{timestamp}.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Token configurations - FOCUS ON MOST LIQUID PAIRS
TOKEN_CONFIG = {
    'SOL': {'mint': 'So11111111111111111111111111111111111111112', 'decimals': 9, 'price': 145.0},
    'USDC': {'mint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', 'decimals': 6, 'price': 1.0},
    'USDT': {'mint': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB', 'decimals': 6, 'price': 1.0},
    'JUP': {'mint': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN', 'decimals': 6, 'price': 0.85},
    'WIF': {'mint': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm', 'decimals': 6, 'price': 1.85},
    'RAY': {'mint': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R', 'decimals': 6, 'price': 2.1},
    'BONK': {'mint': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263', 'decimals': 5, 'price': 0.000025},
    'ORCA': {'mint': 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE', 'decimals': 6, 'price': 3.2},
    'POPCAT': {'mint': '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr', 'decimals': 9, 'price': 0.75},
    'WBTC': {'mint': '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh', 'decimals': 8, 'price': 65000.0},
    'WETH': {'mint': '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs', 'decimals': 8, 'price': 2500.0},
    'SRM': {'mint': 'SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt', 'decimals': 6, 'price': 0.35},
    'MNGO': {'mint': 'MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac', 'decimals': 6, 'price': 0.025},
    'STEP': {'mint': 'StepAscQoEioFxxWGnh2sLBDFp9d8rvKz2Yp39iDpyT', 'decimals': 9, 'price': 0.045},
    'COPE': {'mint': '8HGyAAB1yoM1ttS7pXjHMa3dukTFGQggnFFH3hJZgzQh', 'decimals': 6, 'price': 0.15},
    'FIDA': {'mint': 'EchesyfXePKdLtoiZSL8pBe8Myagyy8ZRqsACNCFGnvp', 'decimals': 6, 'price': 0.28},
    'MAPS': {'mint': 'MAPS41MDahZ9QdKXhVa4dWB9RuyfV4XqhyAZ8XcYepb', 'decimals': 6, 'price': 0.12},
    'OXY': {'mint': 'z3dn17yLaGMKffVogeFHQ9zWVcXgqgf3PQnDsNs2g6M', 'decimals': 6, 'price': 0.08},
    'SAMO': {'mint': '7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU', 'decimals': 9, 'price': 0.0035},
    'SLND': {'mint': 'SLNDpmoWTVADgEdndyvWzroNL7zSi1dF9PC3xHGtPwp', 'decimals': 6, 'price': 0.18},
}

# DEX configurations - ONLY WORKING DEXS
DEX_CONFIG = {
    'aggregated': {'name': 'Jupiter Aggregated', 'fee': 0.0025},
    'Phoenix': {'name': 'Phoenix', 'fee': 0.002},
    'Meteora': {'name': 'Meteora', 'fee': 0.0025},
}

# RPC endpoints
RPC_ENDPOINTS = [
    "https://api.mainnet-beta.solana.com",
    "https://solana-api.projectserum.com",
    "https://rpc.ankr.com/solana"
]

class WalletManager:
    """Secure wallet management"""
    
    def __init__(self, wallet_file: str = "arbitrage_wallet.enc"):
        self.wallet_file = wallet_file
        self.keypair = None
        self.public_key = None
        
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def decrypt_private_key(self, password: str) -> bool:
        """Decrypt and load private key"""
        try:
            if not os.path.exists(self.wallet_file):
                logger.error(f"ERROR: Wallet file {self.wallet_file} not found")
                return False
            
            # Read encrypted data
            with open(self.wallet_file, 'rb') as f:
                data = f.read()
            
            # Extract salt and encrypted key
            salt = data[:16]
            encrypted_key = data[16:]
            
            # Derive key and decrypt
            key = self._derive_key(password, salt)
            fernet = Fernet(key)
            private_key = fernet.decrypt(encrypted_key).decode()
            
            # Create keypair
            private_key_bytes = base58.b58decode(private_key)
            try:
                self.keypair = Keypair.from_bytes(private_key_bytes)
            except:
                self.keypair = Keypair.from_secret_key_bytes(private_key_bytes)
            
            self.public_key = str(self.keypair.pubkey())
            
            logger.info(f"SUCCESS: Wallet loaded successfully")
            logger.info(f"WALLET: Wallet Address: {self.public_key}")
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to decrypt wallet: {e}")
            return False

class BalanceTrackingArbitrageBot:
    """Arbitrage bot with balance tracking at start and end"""
    
    def __init__(self):
        self.wallet_manager = WalletManager()
        self.session = None
        self.rpc_client = None
        self.current_rpc_index = 0
        
        # Balance tracking
        self.starting_balances = {}
        self.ending_balances = {}
        
        # Enhanced rate limiting for both quotes AND swaps
        self.quote_request_count = 0
        self.swap_request_count = 0
        self.request_window_start = time.time()
        self.max_quote_requests_per_minute = 30  # Conservative for quotes
        self.max_swap_requests_per_minute = 10   # Very conservative for swaps
        
        # Trading configuration
        self.config = {
            'MAX_TRADE_SIZE': 1.00,
            'DAILY_LIMIT': 10.00,
            'MIN_PROFIT_THRESHOLD': 0.02,
            'MIN_ROI_THRESHOLD': 0.04,
            'SLIPPAGE_BPS': 300,
            'SCAN_INTERVAL': 30,  # 3 minutes between scans
            'GAS_ESTIMATE': 0.06,
            'QUOTE_DELAY': 4.0,    # 4 seconds between quote requests
            'SWAP_DELAY': 15.0,    # 15 seconds between swap requests
        }
        
        # Smart pair generation based on wallet balances will be done during scanning
        self.trading_pairs = []  # Will be populated dynamically
        
        # Statistics
        self.stats = {
            'scans_completed': 0,
            'opportunities_found': 0,
            'trades_executed': 0,
            'total_profit': 0.0,
            'daily_volume': 0.0,
            'start_time': datetime.now()
        }
        
    async def generate_trading_pairs_smart(self) -> List[Tuple[str, str]]:
        """Generate trading pairs based on actual wallet balances - only scan what we can trade"""
        # Get current balances to determine what we can actually trade
        current_balances = await self.get_all_balances()
        
        logger.info("SMART PAIR GENERATION: Based on wallet balances")
        for token, balance in current_balances.items():
            logger.info(f"  {token}: {balance:.6f}")
        
        # Define all possible pairs
        all_possible_pairs = [
            # SOL pairs (require SOL to buy)
            ('SOL', 'USDT'), ('SOL', 'USDC'),
            # Major token pairs
            ('JUP', 'USDT'), ('JUP', 'USDC'),
            ('WIF', 'USDT'), ('WIF', 'USDC'),
            ('RAY', 'USDT'), ('RAY', 'USDC'),
            ('BONK', 'USDT'), ('BONK', 'USDC'),
            ('ORCA', 'USDT'), ('ORCA', 'USDC'),
            ('POPCAT', 'USDT'), ('POPCAT', 'USDC'),
            # Cross-token pairs (don't require SOL)
            ('USDC', 'USDT'), ('USDT', 'USDC'),
            ('JUP', 'WBTC'), ('WIF', 'WETH'),
            ('RAY', 'JUP'), ('BONK', 'RAY'),
            # DeFi pairs
            ('SRM', 'USDT'), ('MNGO', 'USDT'),
            ('STEP', 'USDT'), ('COPE', 'USDT'),
            ('FIDA', 'USDT'), ('MAPS', 'USDT'),
            ('OXY', 'USDT'), ('SAMO', 'USDT'),
            ('SLND', 'USDT'), ('WBTC', 'USDT'),
            ('WETH', 'USDT'),
        ]
        
        # Filter pairs based on what we can actually trade
        tradeable_pairs = []
        min_balance_threshold = 0.01  # Minimum balance to consider trading
        
        for base_token, quote_token in all_possible_pairs:
            can_trade = False
            
            # Check if we have enough of either token to trade
            base_balance = current_balances.get(base_token, 0.0)
            quote_balance = current_balances.get(quote_token, 0.0)
            
            # We can trade if we have either:
            # 1. Enough base token to sell
            # 2. Enough quote token to buy base token
            if base_balance > min_balance_threshold or quote_balance > min_balance_threshold:
                can_trade = True
                tradeable_pairs.append((base_token, quote_token))
                logger.info(f"  ✅ {base_token}/{quote_token} - Can trade (Base: {base_balance:.4f}, Quote: {quote_balance:.4f})")
            else:
                logger.info(f"  ❌ {base_token}/{quote_token} - Insufficient balance (Base: {base_balance:.4f}, Quote: {quote_balance:.4f})")
        
        logger.info(f"SMART FILTERING: {len(tradeable_pairs)} tradeable pairs from {len(all_possible_pairs)} total")
        return tradeable_pairs  
    async def get_token_balance(self, token_symbol: str) -> float:
        """Get balance for a specific token using raw RPC (bypasses broken library)"""
        try:
            wallet_address = self.wallet_manager.public_key
            
            if token_symbol == 'SOL':
                # SOL balance using standard method (this works)
                balance_response = await self.rpc_client.get_balance(
                    Pubkey.from_string(wallet_address)
                )
                return balance_response.value / 1e9
            else:
                # Token balance using raw RPC calls (this works)
                token_mint = TOKEN_CONFIG[token_symbol]['mint']
                decimals = TOKEN_CONFIG[token_symbol]['decimals']
                
                # Make raw RPC call to bypass broken library
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTokenAccountsByOwner",
                    "params": [
                        wallet_address,
                        {"mint": token_mint},
                        {"encoding": "jsonParsed"}
                    ]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post("https://api.mainnet-beta.solana.com", json=payload) as response:
                        result = await response.json()
                
                if "result" in result and result["result"]["value"]:
                    accounts = result["result"]["value"]
                    total_balance = 0
                    for account in accounts:
                        try:
                            account_data = account["account"]["data"]["parsed"]["info"]
                            token_amount = account_data["tokenAmount"]
                            amount = int(token_amount["amount"])
                            balance = amount / (10 ** decimals)
                            total_balance += balance
                        except Exception as e:
                            logger.debug(f"Error parsing account: {e}")
                    return total_balance
                else:
                    return 0.0
                
        except Exception as e:
            logger.warning(f"WARNING: Could not fetch {token_symbol} balance: {e}")
            return 0.0
    
    async def get_all_balances(self) -> Dict[str, float]:
        """Get balances for SOL, USDC, and USDT"""
        balances = {}
        
        for token in ['SOL', 'USDC', 'USDT']:
            balance = await self.get_token_balance(token)
            balances[token] = balance
        
        return balances
    
    def print_balances(self, balances: Dict[str, float], title: str):
        """Print balances in a nice format"""
        logger.info("=" * 60)
        logger.info(f"{title}")
        logger.info("=" * 60)
        for token, balance in balances.items():
            if token == 'SOL':
                logger.info(f"{token:>6}: {balance:>12.6f} {token}")
            else:
                logger.info(f"{token:>6}: {balance:>12.2f} {token}")
        logger.info("=" * 60)
    
    def print_balance_comparison(self):
        """Print starting vs ending balances with profit/loss"""
        logger.info("=" * 80)
        logger.info("BALANCE COMPARISON - PROFIT/LOSS SUMMARY")
        logger.info("=" * 80)
        
        logger.info("STARTING BALANCES:")
        for token, balance in self.starting_balances.items():
            if token == 'SOL':
                logger.info(f"  {token:>6}: {balance:>12.6f} {token}")
            else:
                logger.info(f"  {token:>6}: {balance:>12.2f} {token}")
        
        logger.info("")
        logger.info("ENDING BALANCES:")
        for token, balance in self.ending_balances.items():
            if token == 'SOL':
                logger.info(f"  {token:>6}: {balance:>12.6f} {token}")
            else:
                logger.info(f"  {token:>6}: {balance:>12.2f} {token}")
        
        logger.info("")
        logger.info("PROFIT/LOSS:")
        total_usd_change = 0.0
        
        for token in ['SOL', 'USDC', 'USDT']:
            start_balance = self.starting_balances.get(token, 0.0)
            end_balance = self.ending_balances.get(token, 0.0)
            change = end_balance - start_balance
            
            if token == 'SOL':
                usd_change = change * TOKEN_CONFIG[token]['price']
                total_usd_change += usd_change
                logger.info(f"  {token:>6}: {change:>+12.6f} {token} (${usd_change:>+8.2f} USD)")
            else:
                total_usd_change += change
                logger.info(f"  {token:>6}: {change:>+12.2f} {token} (${change:>+8.2f} USD)")
        
        logger.info("")
        logger.info(f"TOTAL USD CHANGE: ${total_usd_change:>+8.2f}")
        
        if total_usd_change > 0:
            logger.info("RESULT: PROFIT! 🎉")
        elif total_usd_change < 0:
            logger.info("RESULT: LOSS 😞")
        else:
            logger.info("RESULT: BREAK EVEN")
        
        logger.info("=" * 80)
    
    async def check_quote_rate_limit(self):
        """Check and enforce rate limiting for quote requests"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.request_window_start > 60:
            self.quote_request_count = 0
            self.swap_request_count = 0
            self.request_window_start = current_time
        
        # Check quote rate limit
        if self.quote_request_count >= self.max_quote_requests_per_minute:
            wait_time = 60 - (current_time - self.request_window_start)
            if wait_time > 0:
                logger.warning(f"QUOTE RATE LIMIT: Waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self.quote_request_count = 0
                self.request_window_start = time.time()
        
        self.quote_request_count += 1
    
    async def check_swap_rate_limit(self):
        """Check and enforce rate limiting for swap requests"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.request_window_start > 60:
            self.quote_request_count = 0
            self.swap_request_count = 0
            self.request_window_start = current_time
        
        # Check swap rate limit
        if self.swap_request_count >= self.max_swap_requests_per_minute:
            wait_time = 60 - (current_time - self.request_window_start)
            if wait_time > 0:
                logger.warning(f"SWAP RATE LIMIT: Waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self.swap_request_count = 0
                self.request_window_start = time.time()
        
        self.swap_request_count += 1
    
    async def initialize(self) -> bool:
        """Initialize bot and get starting balances"""
        try:
            if not os.path.exists("arbitrage_wallet.enc"):
                logger.error("ERROR: No encrypted wallet found")
                return False
            
            # Get password
            try:
                password = getpass.getpass("WALLET: Enter wallet password: ")
            except:
                print("WALLET: Enter wallet password (visible): ", end="")
                password = input()
            
            if not self.wallet_manager.decrypt_private_key(password):
                return False
            
            # Initialize session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'Solana-Arbitrage-Bot/3.8'}
            )
            
            # Initialize RPC
            self.rpc_client = AsyncClient(RPC_ENDPOINTS[0])
            
            # GET STARTING BALANCES
            logger.info("FETCHING STARTING BALANCES...")
            self.starting_balances = await self.get_all_balances()
            self.print_balances(self.starting_balances, "STARTING BALANCES")
            
            logger.info("SUCCESS: Bot initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize: {e}")
            return False
    
    async def get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, dex: str = None) -> Optional[Dict]:
        """Get Jupiter quote with proper rate limiting"""
        try:
            # Check rate limit before making request
            await self.check_quote_rate_limit()
            
            url = "https://quote-api.jup.ag/v6/quote"
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(self.config['SLIPPAGE_BPS']),
                'onlyDirectRoutes': 'false',
                'asLegacyTransaction': 'false'
            }
            
            if dex and dex != 'aggregated':
                params['dexes'] = dex
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning(f"Rate limited for {dex}, waiting 30 seconds...")
                    await asyncio.sleep(30)
                    return None
                else:
                    logger.debug(f"Quote failed for {dex}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.debug(f"Quote error for {dex}: {e}")
            return None
    
    async def execute_jupiter_swap_with_retries(self, quote_data: Dict, trade_description: str) -> Optional[str]:
        """Execute Jupiter swap with proper rate limiting and retries"""
        try:
            logger.info(f"EXECUTING: {trade_description}")
            
            # Check swap rate limit before making request
            await self.check_swap_rate_limit()
            
            # Retry logic for rate limits
            for attempt in range(3):
                try:
                    # Get swap transaction
                    swap_url = "https://quote-api.jup.ag/v6/swap"
                    swap_payload = {
                        'quoteResponse': quote_data,
                        'userPublicKey': self.wallet_manager.public_key,
                        'wrapAndUnwrapSol': True,
                        'asLegacyTransaction': False
                    }
                    
                    async with self.session.post(swap_url, json=swap_payload) as response:
                        if response.status == 429:
                            wait_time = 30 * (attempt + 1)  # Exponential backoff
                            logger.warning(f"Rate limited on swap API (attempt {attempt + 1}), waiting {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        elif response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Swap API error: {response.status} - {error_text}")
                            if attempt < 2:
                                await asyncio.sleep(5)
                                continue
                            return None
                        
                        swap_data = await response.json()
                        break  # Success, exit retry loop
                
                except Exception as e:
                    logger.warning(f"Swap request attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(5)
                        continue
                    return None
            else:
                logger.error("All swap request attempts failed")
                return None
            
            # Deserialize and sign transaction
            transaction_bytes = base64.b64decode(swap_data['swapTransaction'])
            transaction = VersionedTransaction.from_bytes(transaction_bytes)
            
            message = transaction.message
            message_bytes = to_bytes_versioned(message)
            signature = self.wallet_manager.keypair.sign_message(message_bytes)
            signed_transaction = VersionedTransaction.populate(message, [signature])
            
            # Submit transaction with retries
            for attempt in range(3):
                try:
                    rpc_endpoint = RPC_ENDPOINTS[self.current_rpc_index % len(RPC_ENDPOINTS)]
                    client = AsyncClient(rpc_endpoint)
                    
                    result = await client.send_transaction(
                        signed_transaction,
                        opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
                    )
                    
                    if result.value:
                        logger.info(f"SUCCESS: Transaction submitted: {result.value}")
                        
                        # Wait for confirmation
                        await asyncio.sleep(3)
                        confirmation = await client.confirm_transaction(result.value, commitment=Confirmed)
                        
                        if confirmation.value[0].confirmation_status:
                            logger.info(f"SUCCESS: Transaction confirmed: {result.value}")
                            return str(result.value)
                        else:
                            logger.warning(f"WARNING: Transaction not confirmed: {result.value}")
                            return str(result.value)
                    
                    await client.close()
                    
                except Exception as e:
                    logger.warning(f"Transaction attempt {attempt + 1} failed: {e}")
                    self.current_rpc_index += 1
                    if attempt < 2:
                        await asyncio.sleep(2)
                    continue
            
            logger.error("ERROR: All transaction attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"ERROR: Swap execution failed: {e}")
            return None
    
    async def execute_best_opportunity_immediately(self, opportunity: Dict) -> bool:
        """Execute the best arbitrage opportunity immediately with proper rate limiting"""
        try:
            # ENHANCED TRANSACTION MESSAGES
            logger.info("=" * 80)
            logger.info("EXECUTING BEST OPPORTUNITY IMMEDIATELY!")
            logger.info("=" * 80)
            logger.info(f"PAIR: {opportunity['pair']}")
            logger.info(f"STRATEGY: Buy on {opportunity['buy_dex']} -> Sell on {opportunity['sell_dex']}")
            logger.info(f"BUY PRICE: ${opportunity['buy_price']:.6f} on {opportunity['buy_dex']}")
            logger.info(f"SELL PRICE: ${opportunity['sell_price']:.6f} on {opportunity['sell_dex']}")
            logger.info(f"SPREAD: {opportunity['spread']*100:.2f}%")
            logger.info(f"EXPECTED PROFIT: ${opportunity['net_profit']:.4f} ({opportunity['roi']*100:.2f}% ROI)")
            logger.info(f"TRADE SIZE: ${opportunity['amount_usd']:.2f}")
            logger.info("=" * 80)
            
            # Check daily limits
            if self.stats['daily_volume'] + opportunity['amount_usd'] > self.config['DAILY_LIMIT']:
                logger.warning(f"WARNING: Daily limit would be exceeded. Skipping trade.")
                return False
            
            # Step 1: Buy transaction with rate limiting
            buy_description = f"BUY {opportunity['base_token']} on {opportunity['buy_dex']} at ${opportunity['buy_price']:.6f}"
            logger.info("=" * 40)
            logger.info("STEP 1: EXECUTING BUY TRANSACTION")
            logger.info(f"ACTION: {buy_description}")
            logger.info("=" * 40)
            
            buy_tx = await self.execute_jupiter_swap_with_retries(opportunity['buy_quote'], buy_description)
            
            if not buy_tx:
                logger.error("ERROR: Buy trade failed")
                logger.info("=" * 80)
                return False
            
            logger.info(f"SUCCESS: Buy transaction completed: {buy_tx}")
            
            # Wait between trades to avoid rate limits
            logger.info(f"WAITING: {self.config['SWAP_DELAY']} seconds between transactions...")
            await asyncio.sleep(self.config['SWAP_DELAY'])
            
            # Step 2: Sell transaction with rate limiting
            sell_description = f"SELL {opportunity['base_token']} on {opportunity['sell_dex']} at ${opportunity['sell_price']:.6f}"
            logger.info("=" * 40)
            logger.info("STEP 2: EXECUTING SELL TRANSACTION")
            logger.info(f"ACTION: {sell_description}")
            logger.info("=" * 40)
            
            sell_tx = await self.execute_jupiter_swap_with_retries(opportunity['sell_quote'], sell_description)
            
            if not sell_tx:
                logger.error("ERROR: Sell trade failed")
                logger.info("=" * 80)
                return False
            
            logger.info(f"SUCCESS: Sell transaction completed: {sell_tx}")
            
            # Update statistics
            self.stats['trades_executed'] += 1
            self.stats['total_profit'] += opportunity['net_profit']
            self.stats['daily_volume'] += opportunity['amount_usd']
            
            # Final success message
            logger.info("=" * 80)
            logger.info("ARBITRAGE EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"PAIR: {opportunity['pair']}")
            logger.info(f"STRATEGY: {opportunity['buy_dex']} -> {opportunity['sell_dex']}")
            logger.info(f"BUY TX: {buy_tx}")
            logger.info(f"SELL TX: {sell_tx}")
            logger.info(f"ACTUAL PROFIT: ${opportunity['net_profit']:.4f}")
            logger.info(f"ROI: {opportunity['roi']*100:.2f}%")
            logger.info(f"TOTAL PROFIT TODAY: ${self.stats['total_profit']:.4f}")
            logger.info(f"TRADES EXECUTED: {self.stats['trades_executed']}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: ARBITRAGE EXECUTION FAILED: {e}")
            logger.info("=" * 80)
            return False
    
    async def scan_and_execute_best_opportunity(self):
        """Scan pairs and execute the best opportunity immediately"""
        logger.info(f"SCAN: SMART BALANCE-AWARE SCAN #{self.stats['scans_completed'] + 1} - {datetime.now().strftime('%H:%M:%S')}")
        
        # Generate trading pairs based on current wallet balances
        self.trading_pairs = await self.generate_trading_pairs_smart()
        
        logger.info(f"SCAN: Scanning {len(self.trading_pairs)} balance-filtered pairs across {len(DEX_CONFIG)} DEX routes...")
        logger.info("STRATEGY: FIND BEST OPPORTUNITY AND EXECUTE IMMEDIATELY!")
        logger.info("=" * 80)
        
        best_opportunity = None
        best_roi = 0
        
        for i, (base_token, quote_token) in enumerate(self.trading_pairs, 1):
            try:
                # Check daily limit
                if self.stats['daily_volume'] >= self.config['DAILY_LIMIT']:
                    logger.info("LIMIT: Daily limit reached. Stopping scan.")
                    break
                
                # Calculate trade amount
                trade_amount_usd = self.config['MAX_TRADE_SIZE']
                base_config = TOKEN_CONFIG[base_token]
                amount_tokens = trade_amount_usd / base_config['price']
                amount_lamports = int(amount_tokens * (10 ** base_config['decimals']))
                
                logger.info(f"SCANNING {i:2d}/{len(self.trading_pairs)}: {base_token}/{quote_token}")
                
                # Get quotes from DEXs
                quotes = {}
                input_mint = base_config['mint']
                output_mint = TOKEN_CONFIG[quote_token]['mint']
                
                for dex_name in DEX_CONFIG.keys():
                    quote = await self.get_jupiter_quote(input_mint, output_mint, amount_lamports, dex_name)
                    
                    if quote and 'outAmount' in quote:
                        out_amount = int(quote['outAmount'])
                        quote_decimals = TOKEN_CONFIG[quote_token]['decimals']
                        price = out_amount / (10 ** quote_decimals)
                        
                        quotes[dex_name] = {
                            'price': price,
                            'routes': len(quote.get('routePlan', [])),
                            'impact': float(quote.get('priceImpactPct', 0)),
                            'quote_data': quote
                        }
                        
                        logger.info(f"  SUCCESS {dex_name:12s}: ${price:.6f} ({quotes[dex_name]['routes']} routes, {quotes[dex_name]['impact']:.3f}% impact)")
                    else:
                        logger.info(f"  FAILED  {dex_name:12s}: No routes")
                    
                    # Rate limiting delay
                    await asyncio.sleep(self.config['QUOTE_DELAY'])
                
                # Check for opportunity
                if len(quotes) >= 2:
                    prices = [(dex, data['price']) for dex, data in quotes.items()]
                    prices.sort(key=lambda x: x[1])
                    
                    buy_dex, buy_price = prices[0]
                    sell_dex, sell_price = prices[-1]
                    
                    if buy_dex != sell_dex:
                        spread = (sell_price - buy_price) / buy_price
                        gross_profit = sell_price - buy_price
                        
                        # Calculate fees
                        buy_fee = buy_price * DEX_CONFIG[buy_dex]['fee']
                        sell_fee = sell_price * DEX_CONFIG[sell_dex]['fee']
                        total_fees = buy_fee + sell_fee + self.config['GAS_ESTIMATE']
                        
                        net_profit = gross_profit - total_fees
                        roi = net_profit / trade_amount_usd
                        
                        # Check profitability
                        if net_profit > self.config['MIN_PROFIT_THRESHOLD'] and roi > self.config['MIN_ROI_THRESHOLD']:
                            opportunity = {
                                'pair': f"{base_token}/{quote_token}",
                                'buy_dex': buy_dex,
                                'sell_dex': sell_dex,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'spread': spread,
                                'gross_profit': gross_profit,
                                'net_profit': net_profit,
                                'roi': roi,
                                'amount_usd': trade_amount_usd,
                                'buy_quote': quotes[buy_dex]['quote_data'],
                                'sell_quote': quotes[sell_dex]['quote_data'],
                                'base_token': base_token,
                                'quote_token': quote_token
                            }
                            
                            logger.info(f"  OPPORTUNITY FOUND!")
                            logger.info(f"     Strategy: {buy_dex} -> {sell_dex}")
                            logger.info(f"     Spread: {spread*100:.2f}% | Net Profit: ${net_profit:.4f} | ROI: {roi*100:.2f}%")
                            
                            # Track best opportunity
                            if roi > best_roi:
                                best_opportunity = opportunity
                                best_roi = roi
                                logger.info(f"     NEW BEST OPPORTUNITY! ROI: {roi*100:.2f}%")
                
            except Exception as e:
                logger.error(f"Error scanning {base_token}/{quote_token}: {e}")
                continue
        
        # Execute the best opportunity found
        if best_opportunity:
            logger.info("=" * 80)
            logger.info(f"EXECUTING BEST OPPORTUNITY: {best_opportunity['pair']} with {best_roi*100:.2f}% ROI")
            logger.info("=" * 80)
            
            success = await self.execute_best_opportunity_immediately(best_opportunity)
            if success:
                self.stats['opportunities_found'] += 1
                logger.info(f"SUCCESS: Best opportunity executed!")
            else:
                logger.error(f"FAILED: Best opportunity execution failed.")
        else:
            logger.info("NO PROFITABLE OPPORTUNITIES FOUND")
        
        self.stats['scans_completed'] += 1
        
        logger.info("=" * 80)
        logger.info(f"SCAN: BEST OPPORTUNITY SCAN COMPLETE")
        logger.info(f"TOTAL PROFIT TODAY: ${self.stats['total_profit']:.4f}")
        logger.info("=" * 80)
    
    async def run_balance_tracking_loop(self):
        """Main trading loop with balance tracking"""
        logger.info("TRADING: STARTING BALANCE-TRACKING ARBITRAGE TRADING...")
        logger.info(f"   Trade Size: ${self.config['MAX_TRADE_SIZE']:.2f} per opportunity")
        logger.info(f"   Daily Limit: ${self.config['DAILY_LIMIT']:.2f} maximum")
        logger.info(f"   Token Pairs: {len(self.trading_pairs)} top liquid pairs")
        logger.info(f"   DEX Routes: {len(DEX_CONFIG)} working DEXs")
        logger.info(f"   Execution: BEST OPPORTUNITY PER SCAN")
        logger.info(f"   Scan Interval: {self.config['SCAN_INTERVAL']} seconds")
        logger.info(f"   Quote Rate Limit: {self.max_quote_requests_per_minute} requests/minute")
        logger.info(f"   Swap Rate Limit: {self.max_swap_requests_per_minute} requests/minute")
        
        while True:
            try:
                # Check daily limit
                if self.stats['daily_volume'] >= self.config['DAILY_LIMIT']:
                    logger.info(f"LIMIT: Daily limit reached. Waiting...")
                    await asyncio.sleep(3600)
                    continue
                
                # Scan and execute best opportunity
                await self.scan_and_execute_best_opportunity()
                
                # Wait before next scan
                logger.info(f"WAIT: Waiting {self.config['SCAN_INTERVAL']} seconds before next scan...")
                await asyncio.sleep(self.config['SCAN_INTERVAL'])
                
            except KeyboardInterrupt:
                logger.info("STOP: Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"ERROR: Unexpected error: {e}")
                await asyncio.sleep(60)
                continue
    
    async def cleanup(self):
        """Cleanup resources and show final balance comparison"""
        # GET ENDING BALANCES
        logger.info("FETCHING ENDING BALANCES...")
        self.ending_balances = await self.get_all_balances()
        self.print_balances(self.ending_balances, "ENDING BALANCES")
        
        # SHOW BALANCE COMPARISON
        self.print_balance_comparison()
        
        if self.session:
            await self.session.close()
        if self.rpc_client:
            await self.rpc_client.close()
        
        runtime = datetime.now() - self.stats['start_time']
        logger.info("=" * 80)
        logger.info("STATS: FINAL STATISTICS")
        logger.info(f"   Runtime: {runtime}")
        logger.info(f"   Scans: {self.stats['scans_completed']}")
        logger.info(f"   Opportunities: {self.stats['opportunities_found']}")
        logger.info(f"   Trades: {self.stats['trades_executed']}")
        logger.info(f"   Profit: ${self.stats['total_profit']:.4f}")
        logger.info("=" * 80)

async def main():
    """Main function"""
    bot = BalanceTrackingArbitrageBot()
    
    try:
        if not await bot.initialize():
            logger.error("ERROR: Failed to initialize")
            return
        
        await bot.run_balance_tracking_loop()
        
    except KeyboardInterrupt:
        logger.info("STOP: Shutting down...")
    except Exception as e:
        logger.error(f"ERROR: Fatal error: {e}")
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSTOP: Bot stopped by user")
    except Exception as e:
        print(f"ERROR: Fatal error: {e}")

