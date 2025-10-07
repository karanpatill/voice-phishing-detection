#!/usr/bin/env python3
"""
Main Integration Script for Voice Phishing Detection System
Integrates with Linphone for real-time call recording and analysis
"""

import sys
import os
import time
import threading
import signal
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from client.linphone_recorder import LinphoneCallRecorder

class VoicePhishingDetectionSystem:
    def __init__(self, backend_url="http://127.0.0.1:8000", chunk_duration=5):
        self.backend_url = backend_url
        self.chunk_duration = chunk_duration
        self.recorder = None
        self.is_running = False
        self.current_call = None
        
        print("🛡️  Voice Phishing Detection System")
        print("=" * 50)
        print(f"Backend URL: {backend_url}")
        print(f"Chunk Duration: {chunk_duration}s")
        print("=" * 50)
    
    def initialize(self):
        """Initialize the system components"""
        try:
            print("🔧 Initializing system components...")
            
            # Initialize Linphone recorder
            self.recorder = LinphoneCallRecorder(
                backend_url=self.backend_url,
                chunk_duration=self.chunk_duration
            )
            
            print("✅ System initialization completed")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def start_monitoring(self, call_id=None):
        """Start monitoring for calls"""
        if not self.recorder:
            print("❌ System not initialized. Call initialize() first.")
            return False
        
        if self.current_call:
            print("⚠️  Already monitoring a call. Stop current monitoring first.")
            return False
        
        try:
            success = self.recorder.start_call_recording(call_id)
            if success:
                self.current_call = self.recorder.current_call_id
                print(f"✅ Monitoring started for call: {self.current_call}")
                return True
            else:
                print("❌ Failed to start monitoring")
                return False
                
        except Exception as e:
            print(f"❌ Error starting monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop current call monitoring"""
        if not self.current_call:
            print("⚠️  No active monitoring to stop")
            return False
        
        try:
            self.recorder.stop_call_recording() # type: ignore
            print(f"✅ Monitoring stopped for call: {self.current_call}")
            self.current_call = None
            return True
            
        except Exception as e:
            print(f"❌ Error stopping monitoring: {e}")
            return False
    
    def get_status(self):
        """Get current system status"""
        if not self.recorder:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "monitoring": self.current_call is not None,
            "current_call": self.current_call,
            "recorder_status": self.recorder.get_status()
        }
    
    def run_interactive_mode(self):
        """Run system in interactive mode"""
        print("\\n🎮 Interactive Mode Started")
        print("Commands:")
        print("  start [call_id] - Start monitoring (optional call_id)")
        print("  stop           - Stop current monitoring")
        print("  status         - Show system status")
        print("  help           - Show this help")
        print("  quit           - Exit system")
        print("\\nType 'help' for commands or 'quit' to exit\\n")
        
        self.is_running = True
        
        while self.is_running:
            try:
                command = input("📞 VPD> ").strip().lower()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0]
                
                if cmd == "start":
                    call_id = parts[1] if len(parts) > 1 else None
                    self.start_monitoring(call_id)
                
                elif cmd == "stop":
                    self.stop_monitoring()
                
                elif cmd == "status":
                    status = self.get_status()
                    self.print_status(status)
                
                elif cmd == "help":
                    self.print_help()
                
                elif cmd in ["quit", "exit", "q"]:
                    print("🛑 Shutting down...")
                    self.shutdown()
                    break
                
                else:
                    print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\\n⏹️  Interrupted by user")
                self.shutdown()
                break
                
            except EOFError:
                print("\\n🛑 End of input")
                self.shutdown()
                break
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_auto_mode(self, duration=None):
        """Run system in automatic mode"""
        print(f"🤖 Automatic Mode Started")
        
        if duration:
            print(f"⏱️  Will run for {duration} seconds")
        
        # Auto-start monitoring
        call_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.start_monitoring(call_id):
            self.is_running = True
            
            try:
                if duration:
                    print(f"🕐 Recording for {duration} seconds...")
                    time.sleep(duration)
                    print("🕐 Duration elapsed, stopping...")
                else:
                    print("🕐 Recording indefinitely. Press Ctrl+C to stop...")
                    while self.is_running:
                        time.sleep(1)
                        
            except KeyboardInterrupt:
                print("\\n⏹️  Interrupted by user")
            
            finally:
                self.stop_monitoring()
        
        self.shutdown()
    
    def print_status(self, status):
        """Print formatted status information"""
        print("\\n📊 System Status:")
        print("-" * 30)
        print(f"System: {status['status']}")
        
        if status['status'] == 'initialized':
            print(f"Monitoring: {'Yes' if status['monitoring'] else 'No'}")
            if status['current_call']:
                print(f"Current Call: {status['current_call']}")
            
            if 'recorder_status' in status:
                rs = status['recorder_status']
                print(f"Recording: {'Yes' if rs['is_recording'] else 'No'}")
                print(f"Chunks Processed: {rs['chunk_number']}")
                print(f"WebSocket: {'Connected' if rs['ws_connected'] else 'Disconnected'}")
                print(f"Buffer Size: {rs['buffer_size']} samples")
        
        print("-" * 30 + "\\n")
    
    def print_help(self):
        """Print help information"""
        print("\\n📖 Available Commands:")
        print("-" * 40)
        print("start [call_id]  - Start call monitoring")
        print("                   Optional call_id (auto-generated if not provided)")
        print("stop             - Stop current call monitoring")
        print("status           - Show detailed system status")
        print("help             - Show this help message")
        print("quit/exit/q      - Exit the system")
        print("-" * 40)
        print("\\n💡 Tips:")
        print("- Start Linphone and make/receive calls")
        print("- System will automatically detect and analyze speech")
        print("- Open http://127.0.0.1:8000/static/realtime.html for web dashboard")
        print("- Phishing alerts will be shown in real-time")
        print("-" * 40 + "\\n")
    
    def shutdown(self):
        """Shutdown the system cleanly"""
        print("🧹 Cleaning up resources...")
        
        self.is_running = False
        
        if self.current_call:
            self.stop_monitoring()
        
        if self.recorder:
            self.recorder.cleanup()
        
        print("✅ System shutdown completed")
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        print(f"\\n🔔 Received signal {signum}")
        self.shutdown()
        sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Voice Phishing Detection System - Linphone Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Interactive mode
  python main.py --auto             # Auto mode (indefinite)
  python main.py --auto --duration 60  # Auto mode (60 seconds)
  python main.py --backend http://localhost:8001  # Custom backend
        """
    )
    
    parser.add_argument(
        '--backend',
        default='http://127.0.0.1:8000',
        help='Backend URL (default: http://127.0.0.1:8000)'
    )
    
    parser.add_argument(
        '--chunk-duration',
        type=int,
        default=10,
        help='Audio chunk duration in seconds (default: 10)'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Run in automatic mode'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        help='Duration for auto mode in seconds (default: indefinite)'
    )
    
    parser.add_argument(
        '--call-id',
        help='Custom call ID for auto mode'
    )
    
    args = parser.parse_args()
    
    # Create system instance
    system = VoicePhishingDetectionSystem(
        backend_url=args.backend,
        chunk_duration=args.chunk_duration
    )
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)
    
    try:
        # Initialize system
        if not system.initialize():
            print("❌ System initialization failed. Exiting.")
            sys.exit(1)
        
        # Run in requested mode
        if args.auto:
            system.run_auto_mode(args.duration)
        else:
            system.run_interactive_mode()
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        system.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()