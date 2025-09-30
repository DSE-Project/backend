import asyncio
import socket
import httpx
import os
from datetime import datetime

async def diagnose_fred_connection():
    """Comprehensive FRED API connection diagnostics"""
    
    print("🔍 FRED API Connection Diagnostics")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    
    # 1. Check FRED API Key
    print("\n1. Environment Variables")
    print("-" * 30)
    fred_api_key = os.getenv('FRED_API_KEY')
    if fred_api_key:
        print(f"✅ FRED_API_KEY is set (length: {len(fred_api_key)})")
        print(f"   First 8 chars: {fred_api_key[:8]}...")
    else:
        print("❌ FRED_API_KEY is not set!")
        return
    
    # 2. DNS Resolution Test
    print("\n2. DNS Resolution Test")
    print("-" * 30)
    try:
        hostname = "api.stlouisfed.org"
        ip_address = socket.gethostbyname(hostname)
        print(f"✅ DNS Resolution successful: {hostname} → {ip_address}")
    except socket.gaierror as e:
        print(f"❌ DNS Resolution failed for {hostname}")
        print(f"   Error: {e}")
        print("\n🔧 Possible Solutions:")
        print("   • Check your internet connection")
        print("   • Try using a different DNS server (8.8.8.8, 1.1.1.1)")
        print("   • Check if you're behind a corporate firewall")
        print("   • Temporarily disable VPN if using one")
        return
    except Exception as e:
        print(f"❌ Unexpected DNS error: {e}")
        return
    
    # 3. Basic HTTP Connection Test
    print("\n3. Basic HTTP Connection Test")
    print("-" * 30)
    try:
        timeout_config = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.get("https://api.stlouisfed.org/")
            print(f"✅ HTTP connection successful (Status: {response.status_code})")
    except httpx.TimeoutException:
        print("❌ HTTP connection timed out")
        print("   • Network may be slow or FRED API is down")
    except httpx.ConnectError as e:
        print(f"❌ HTTP connection failed: {e}")
        print("   • Check firewall settings")
        print("   • Verify internet connectivity")
    except Exception as e:
        print(f"❌ Unexpected HTTP error: {e}")
        return
    
    # 4. FRED API Endpoint Test
    print("\n4. FRED API Endpoint Test")
    print("-" * 30)
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "FEDFUNDS",
            "api_key": fred_api_key,
            "file_type": "json",
            "limit": 1,
            "sort_order": "desc"
        }
        
        timeout_config = httpx.Timeout(15.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if "observations" in data and len(data["observations"]) > 0:
                    obs = data["observations"][0]
                    print(f"✅ FRED API working correctly")
                    print(f"   Latest FEDFUNDS: {obs['value']} on {obs['date']}")
                else:
                    print(f"⚠️ FRED API returned unexpected format: {data}")
            elif response.status_code == 400:
                print("❌ FRED API returned 400 - Check API key validity")
                print(f"   Response: {response.text}")
            elif response.status_code == 429:
                print("⚠️ FRED API rate limit exceeded - wait and try again")
            else:
                print(f"❌ FRED API returned status {response.status_code}")
                print(f"   Response: {response.text}")
                
    except httpx.TimeoutException:
        print("❌ FRED API request timed out")
    except httpx.ConnectError as e:
        print(f"❌ FRED API connection failed: {e}")
    except Exception as e:
        print(f"❌ FRED API test failed: {e}")
        return
    
    # 5. Network Configuration Info
    print("\n5. Network Configuration")
    print("-" * 30)
    try:
        # Get default gateway/DNS info if possible
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            result = subprocess.run(["ipconfig", "/all"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                dns_servers = [line.strip() for line in lines if 'DNS Servers' in line]
                if dns_servers:
                    print(f"📡 DNS Servers: {dns_servers[0]}")
                else:
                    print("📡 DNS server info not found in ipconfig output")
        else:
            # For Unix-like systems
            try:
                with open('/etc/resolv.conf', 'r') as f:
                    resolv_content = f.read()
                    dns_lines = [line for line in resolv_content.split('\n') if line.startswith('nameserver')]
                    if dns_lines:
                        print(f"📡 DNS Servers: {', '.join(dns_lines)}")
            except:
                print("📡 Could not read DNS configuration")
                
    except Exception as e:
        print(f"📡 Could not get network info: {e}")
    
    # 6. Recommendations
    print("\n6. Troubleshooting Recommendations")
    print("-" * 30)
    print("If you're still having issues, try:")
    print("1. 🌐 Check internet connectivity: ping google.com")
    print("2. 🔒 Disable VPN temporarily")
    print("3. 🛡️ Check Windows Firewall/antivirus settings")
    print("4. 🏢 If on corporate network, ask IT about FRED API access")
    print("5. 📡 Try changing DNS to 8.8.8.8 or 1.1.1.1")
    print("6. 🔄 Restart network adapter/router")
    print("7. ⏰ Wait a few minutes and try again (temporary outage)")

if __name__ == "__main__":
    asyncio.run(diagnose_fred_connection())