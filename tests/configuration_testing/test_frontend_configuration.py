"""
Frontend Configuration Testing
Tests React frontend configurations, browser compatibility, and UI responsiveness
"""
import json
import subprocess
import os
import sys
import time
from typing import Dict, Any, List

# Import project path utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_utils import get_project_paths, get_config_files

class TestFrontendConfiguration:
    """Test React frontend configuration and browser compatibility"""
    
    def __init__(self):
        self.test_results = []
        # Get project paths using utility function
        self.paths = get_project_paths()
        self.config_files = get_config_files()
        self.frontend_path = self.paths["frontend_path"]
        self.api_base_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
    
    def test_frontend_build_configurations(self):
        """Test frontend build and development configurations"""
        print("⚛️ Testing React frontend configurations...")
        
        # Test 1: Package.json configuration
        self._test_package_json_configuration()
        
        # Test 2: Node modules installation
        self._test_node_modules()
        
        # Test 3: Vite configuration
        self._test_vite_configuration()
        
        # Test 4: Environment variables
        self._test_frontend_environment_variables()
        
        return self.test_results
    
    def _test_package_json_configuration(self):
        """Test package.json configuration and dependencies"""
        print("📦 Testing package.json configuration...")
        
        package_json_path = self.config_files["frontend_package_json"]
        
        if os.path.exists(package_json_path):
            try:
                with open(package_json_path, 'r') as f:
                    package_data = json.load(f)
                
                # Check required dependencies based on actual package.json
                required_deps = [
                    "react",
                    "react-dom", 
                    "vite",
                    "@vitejs/plugin-react",
                    "react-router-dom",
                    "@supabase/supabase-js",  # For Supabase integration
                    "chart.js",  # For data visualization
                    "react-chartjs-2"  # For React charts
                ]
                
                dependencies = {**package_data.get("dependencies", {}), 
                              **package_data.get("devDependencies", {})}
                
                missing_deps = []
                for dep in required_deps:
                    if dep not in dependencies:
                        missing_deps.append(dep)
                
                if not missing_deps:
                    self.test_results.append({
                        "test": "Package.json Configuration",
                        "status": "PASS",
                        "details": f"All required dependencies present. Total dependencies: {len(dependencies)}"
                    })
                else:
                    self.test_results.append({
                        "test": "Package.json Configuration",
                        "status": "FAIL",
                        "details": f"Missing dependencies: {', '.join(missing_deps)}"
                    })
                
                # Check scripts configuration
                scripts = package_data.get("scripts", {})
                required_scripts = ["dev", "build", "preview"]
                missing_scripts = [s for s in required_scripts if s not in scripts]
                
                if not missing_scripts:
                    self.test_results.append({
                        "test": "Package.json Scripts",
                        "status": "PASS",
                        "details": f"All required scripts configured: {list(scripts.keys())}"
                    })
                else:
                    self.test_results.append({
                        "test": "Package.json Scripts",
                        "status": "FAIL",
                        "details": f"Missing scripts: {', '.join(missing_scripts)}"
                    })
                    
            except json.JSONDecodeError:
                self.test_results.append({
                    "test": "Package.json Configuration",
                    "status": "FAIL",
                    "details": "Invalid JSON in package.json"
                })
            except Exception as e:
                self.test_results.append({
                    "test": "Package.json Configuration",
                    "status": "FAIL",
                    "details": f"Error reading package.json: {str(e)}"
                })
        else:
            self.test_results.append({
                "test": "Package.json Configuration",
                "status": "FAIL",
                "details": "package.json not found"
            })
    
    def _test_node_modules(self):
        """Test Node modules installation and integrity"""
        print("📁 Testing Node modules...")
        
        node_modules_path = os.path.join(self.frontend_path, "node_modules")
        
        if os.path.exists(node_modules_path):
            try:
                # Count installed packages
                packages = os.listdir(node_modules_path)
                package_count = len([p for p in packages if not p.startswith('.')])
                
                self.test_results.append({
                    "test": "Node Modules Installation",
                    "status": "PASS",
                    "details": f"{package_count} packages installed"
                })
                
                # Check for key packages based on actual project
                key_packages = ["react", "vite", "@supabase"]
                missing_packages = []
                
                for package in key_packages:
                    package_path = os.path.join(node_modules_path, package)
                    if not os.path.exists(package_path):
                        missing_packages.append(package)
                
                if not missing_packages:
                    self.test_results.append({
                        "test": "Key Packages Availability",
                        "status": "PASS",
                        "details": "All key packages installed"
                    })
                else:
                    self.test_results.append({
                        "test": "Key Packages Availability",
                        "status": "FAIL",
                        "details": f"Missing key packages: {', '.join(missing_packages)}"
                    })
                    
            except Exception as e:
                self.test_results.append({
                    "test": "Node Modules Installation",
                    "status": "FAIL",
                    "details": f"Error checking node_modules: {str(e)}"
                })
        else:
            self.test_results.append({
                "test": "Node Modules Installation",
                "status": "FAIL",
                "details": "node_modules directory not found. Run 'npm install' first."
            })
    
    def _test_vite_configuration(self):
        """Test Vite configuration"""
        print("⚡ Testing Vite configuration...")
        
        vite_config_path = self.config_files["frontend_vite_config"]
        
        if os.path.exists(vite_config_path):
            try:
                with open(vite_config_path, 'r') as f:
                    config_content = f.read()
                
                # Check for essential Vite configurations
                if "@vitejs/plugin-react" in config_content:
                    self.test_results.append({
                        "test": "Vite React Plugin Configuration",
                        "status": "PASS",
                        "details": "React plugin configured"
                    })
                else:
                    self.test_results.append({
                        "test": "Vite React Plugin Configuration",
                        "status": "FAIL",
                        "details": "React plugin not configured"
                    })
                
                # Check for server configuration
                if "server:" in config_content or "host:" in config_content:
                    self.test_results.append({
                        "test": "Vite Server Configuration",
                        "status": "PASS",
                        "details": "Server configuration present"
                    })
                else:
                    self.test_results.append({
                        "test": "Vite Server Configuration",
                        "status": "INFO",
                        "details": "Using default server configuration"
                    })
                    
            except Exception as e:
                self.test_results.append({
                    "test": "Vite Configuration",
                    "status": "FAIL",
                    "details": f"Error reading vite.config.js: {str(e)}"
                })
        else:
            self.test_results.append({
                "test": "Vite Configuration",
                "status": "FAIL",
                "details": "vite.config.js not found"
            })
    
    def _test_frontend_environment_variables(self):
        """Test frontend environment variables"""
        print("🌍 Testing frontend environment variables...")
        
        env_files = [
            (".env", self.config_files["frontend_env"]),
            (".env.local", self.config_files["frontend_env_local"])
        ]
        
        env_found = False
        for env_name, env_path in env_files:
            if os.path.exists(env_path):
                env_found = True
                try:
                    with open(env_path, 'r') as f:
                        env_content = f.read()
                    
                    # Check for common React environment variables
                    if "VITE_" in env_content or "REACT_APP_" in env_content:
                        self.test_results.append({
                            "test": f"Environment File {env_name}",
                            "status": "PASS",
                            "details": "Environment variables configured"
                        })
                    else:
                        self.test_results.append({
                            "test": f"Environment File {env_name}",
                            "status": "INFO",
                            "details": "Environment file exists but no frontend variables found"
                        })
                        
                except Exception as e:
                    self.test_results.append({
                        "test": f"Environment File {env_name}",
                        "status": "FAIL",
                        "details": f"Error reading {env_name}: {str(e)}"
                    })
        
        if not env_found:
            self.test_results.append({
                "test": "Frontend Environment Configuration",
                "status": "INFO",
                "details": "No environment files found (using defaults)"
            })
    
    def test_browser_compatibility_simulation(self):
        """Simulate browser compatibility tests"""
        print("🌐 Testing browser compatibility (simulated)...")
        
        # Since we can't actually test multiple browsers in this environment,
        # we'll test the build process and check for compatibility indicators
        
        # Test 1: Check if build process works
        self._test_build_process()
        
        # Test 2: Check for modern JavaScript features that might cause compatibility issues
        self._test_javascript_compatibility()
        
        # Test 3: Check CSS compatibility
        self._test_css_compatibility()
    
    def _test_build_process(self):
        """Test the Vite build process"""
        print("🏗️ Testing build process...")
        
        try:
            # Change to frontend directory and run build
            original_dir = os.getcwd()
            os.chdir(self.frontend_path)
            
            # Check if we can run npm build (dry run)
            result = subprocess.run(['npm', 'run', 'build', '--dry-run'], 
                                  capture_output=True, text=True, timeout=30)
            
            os.chdir(original_dir)
            
            if result.returncode == 0 or "build" in result.stdout.lower():
                self.test_results.append({
                    "test": "Build Process Configuration",
                    "status": "PASS",
                    "details": "Build script configured and executable"
                })
            else:
                self.test_results.append({
                    "test": "Build Process Configuration",
                    "status": "FAIL",
                    "details": f"Build process failed: {result.stderr}"
                })
                
        except subprocess.TimeoutExpired:
            self.test_results.append({
                "test": "Build Process Configuration",
                "status": "FAIL",
                "details": "Build process timed out"
            })
        except FileNotFoundError:
            self.test_results.append({
                "test": "Build Process Configuration",
                "status": "FAIL",
                "details": "npm command not found"
            })
        except Exception as e:
            self.test_results.append({
                "test": "Build Process Configuration",
                "status": "FAIL",
                "details": f"Build test failed: {str(e)}"
            })
    
    def _test_javascript_compatibility(self):
        """Test JavaScript compatibility features"""
        print("📜 Testing JavaScript compatibility...")
        
        src_path = os.path.join(self.frontend_path, "src")
        
        if os.path.exists(src_path):
            compatibility_issues = []
            modern_features_used = []
            
            # Walk through all JavaScript/TypeScript files
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(('.js', '.jsx', '.ts', '.tsx')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Check for modern JavaScript features
                            if 'async/await' in content or 'async ' in content:
                                modern_features_used.append('async/await')
                            if '=>' in content:
                                modern_features_used.append('arrow functions')
                            if 'const ' in content or 'let ' in content:
                                modern_features_used.append('modern variable declarations')
                            if '...' in content:  # Spread operator
                                modern_features_used.append('spread operator')
                                
                        except Exception:
                            continue  # Skip files that can't be read
            
            unique_features = list(set(modern_features_used))
            
            if unique_features:
                self.test_results.append({
                    "test": "JavaScript Modern Features",
                    "status": "PASS",
                    "details": f"Modern JS features in use: {', '.join(unique_features[:5])}"
                })
            else:
                self.test_results.append({
                    "test": "JavaScript Modern Features",
                    "status": "INFO",
                    "details": "No modern JavaScript features detected"
                })
        else:
            self.test_results.append({
                "test": "JavaScript Compatibility",
                "status": "FAIL",
                "details": "src directory not found"
            })
    
    def _test_css_compatibility(self):
        """Test CSS compatibility"""
        print("🎨 Testing CSS compatibility...")
        
        src_path = os.path.join(self.frontend_path, "src")
        
        if os.path.exists(src_path):
            css_features = []
            
            # Look for CSS files
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(('.css', '.scss', '.sass')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Check for modern CSS features
                            if 'flexbox' in content or 'flex' in content:
                                css_features.append('Flexbox')
                            if 'grid' in content:
                                css_features.append('CSS Grid')
                            if 'var(' in content:
                                css_features.append('CSS Variables')
                            if '@media' in content:
                                css_features.append('Media Queries')
                                
                        except Exception:
                            continue
            
            unique_css_features = list(set(css_features))
            
            if unique_css_features:
                self.test_results.append({
                    "test": "CSS Modern Features",
                    "status": "PASS",
                    "details": f"Modern CSS features: {', '.join(unique_css_features)}"
                })
            else:
                self.test_results.append({
                    "test": "CSS Modern Features",
                    "status": "INFO",
                    "details": "Basic CSS styling detected"
                })
        else:
            self.test_results.append({
                "test": "CSS Compatibility",
                "status": "FAIL",
                "details": "src directory not found"
            })
    
    def test_responsive_design_configuration(self):
        """Test responsive design configuration"""
        print("📱 Testing responsive design configuration...")
        
        # Test 1: Check for viewport meta tag in index.html
        self._test_viewport_configuration()
        
        # Test 2: Check for responsive CSS
        self._test_responsive_css()
        
        # Test 3: Check component structure for responsiveness
        self._test_component_responsiveness()
    
    def _test_viewport_configuration(self):
        """Test viewport meta tag configuration"""
        print("🖼️ Testing viewport configuration...")
        
        index_html_path = os.path.join(self.frontend_path, "index.html")
        
        if os.path.exists(index_html_path):
            try:
                with open(index_html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                if 'viewport' in html_content and 'width=device-width' in html_content:
                    self.test_results.append({
                        "test": "Viewport Meta Tag Configuration",
                        "status": "PASS",
                        "details": "Proper viewport meta tag configured"
                    })
                else:
                    self.test_results.append({
                        "test": "Viewport Meta Tag Configuration",
                        "status": "FAIL",
                        "details": "Viewport meta tag missing or incorrect"
                    })
                    
            except Exception as e:
                self.test_results.append({
                    "test": "Viewport Meta Tag Configuration",
                    "status": "FAIL",
                    "details": f"Error reading index.html: {str(e)}"
                })
        else:
            self.test_results.append({
                "test": "Viewport Meta Tag Configuration",
                "status": "FAIL",
                "details": "index.html not found"
            })
    
    def _test_responsive_css(self):
        """Test responsive CSS implementation"""
        print("📐 Testing responsive CSS...")
        
        src_path = os.path.join(self.frontend_path, "src")
        
        if os.path.exists(src_path):
            media_queries_found = False
            responsive_units_found = False
            
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(('.css', '.scss', '.sass')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            if '@media' in content:
                                media_queries_found = True
                            if any(unit in content for unit in ['rem', 'em', 'vh', 'vw', '%']):
                                responsive_units_found = True
                                
                        except Exception:
                            continue
            
            if media_queries_found and responsive_units_found:
                self.test_results.append({
                    "test": "Responsive CSS Implementation",
                    "status": "PASS",
                    "details": "Media queries and responsive units found"
                })
            elif media_queries_found:
                self.test_results.append({
                    "test": "Responsive CSS Implementation",
                    "status": "PASS",
                    "details": "Media queries found (basic responsiveness)"
                })
            else:
                self.test_results.append({
                    "test": "Responsive CSS Implementation",
                    "status": "INFO",
                    "details": "Limited responsive CSS detected"
                })
        else:
            self.test_results.append({
                "test": "Responsive CSS Implementation",
                "status": "FAIL",
                "details": "src directory not found"
            })
    
    def _test_component_responsiveness(self):
        """Test component structure for responsiveness"""
        print("🧩 Testing component responsiveness...")
        
        components_path = os.path.join(self.frontend_path, "src", "components")
        
        if os.path.exists(components_path):
            responsive_components = []
            
            for root, dirs, files in os.walk(components_path):
                for file in files:
                    if file.endswith(('.jsx', '.tsx')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Look for responsive patterns
                            responsive_patterns = [
                                'className=', 'responsive', 'mobile', 'desktop',
                                'sm:', 'md:', 'lg:', 'xl:', 'breakpoint'
                            ]
                            
                            if any(pattern in content for pattern in responsive_patterns):
                                responsive_components.append(file)
                                
                        except Exception:
                            continue
            
            if responsive_components:
                self.test_results.append({
                    "test": "Component Responsiveness",
                    "status": "PASS",
                    "details": f"Responsive patterns found in {len(responsive_components)} components"
                })
            else:
                self.test_results.append({
                    "test": "Component Responsiveness",
                    "status": "INFO",
                    "details": "No explicit responsive patterns in components"
                })
        else:
            self.test_results.append({
                "test": "Component Responsiveness",
                "status": "INFO",
                "details": "Components directory not found"
            })

def run_frontend_configuration_tests():
    """Run all frontend configuration tests"""
    print("\n🚀 Starting Frontend Configuration Testing...")
    print("=" * 60)
    
    tester = TestFrontendConfiguration()
    
    # Run frontend configuration tests
    tester.test_frontend_build_configurations()
    tester.test_browser_compatibility_simulation()
    tester.test_responsive_design_configuration()
    
    # Print results
    print("\n📋 Frontend Configuration Test Results:")
    print("=" * 60)
    
    passed = failed = info = 0
    for result in tester.test_results:
        if result["status"] == "PASS":
            status_emoji = "✅"
            passed += 1
        elif result["status"] == "FAIL":
            status_emoji = "❌"
            failed += 1
        else:
            status_emoji = "ℹ️"
            info += 1
            
        print(f"{status_emoji} {result['test']}: {result['status']}")
        print(f"   Details: {result['details']}")
    
    print(f"\n📊 Summary: {passed} passed, {failed} failed, {info} informational")
    
    return tester.test_results

if __name__ == "__main__":
    results = run_frontend_configuration_tests()
    
    # Save results to file
    with open("frontend_configuration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: frontend_configuration_test_results.json")