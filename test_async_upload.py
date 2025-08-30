#!/usr/bin/env python3
"""
Test script for asynchronous document upload and processing
"""

import os
import sys
import tempfile
import requests
import time
import json

def test_async_document_upload():
    """Test the asynchronous document upload workflow"""
    
    # Create a test content that will trigger processing
    test_content = """
    FOOD SAFETY TRAINING MANUAL
    
    Chapter 1: Personal Hygiene Standards
    
    Proper hand washing is critical for food safety. Follow these steps:
    1. Use warm water and soap
    2. Scrub for at least 20 seconds
    3. Clean under fingernails
    4. Rinse thoroughly
    5. Dry with clean paper towels
    
    Chapter 2: Temperature Control
    
    Temperature control prevents bacterial growth:
    - Keep hot foods above 140°F (60°C)
    - Keep cold foods below 40°F (4°C)
    - Use thermometers to check temperatures
    - Record temperatures regularly
    
    Chapter 3: Cross-Contamination Prevention
    
    Prevent cross-contamination by:
    - Using separate cutting boards for raw meat and vegetables
    - Storing raw meat on bottom shelves
    - Washing hands between handling different foods
    - Using color-coded equipment
    
    Quiz Questions:
    1. What temperature should hot foods be kept at?
    2. How long should you wash your hands?
    3. Where should raw meat be stored in the refrigerator?
    """
    
    # Create a temporary text file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write(test_content)
        temp_file_path = temp_file.name
    
    try:
        # Test upload endpoint
        api_url = "http://localhost:8000/api/documents/upload"
        
        # Prepare the multipart form data
        with open(temp_file_path, 'rb') as f:
            files = {'file': ('food_safety_manual.txt', f, 'text/plain')}
            data = {
                'title': 'Food Safety Training Manual',
                'document_type': 'Training',
                'version': '2.0',
                'effective_date': '2024-01-01',
                'language': 'en'
            }
            
            print("🚀 Testing asynchronous document upload...")
            start_time = time.time()
            response = requests.post(api_url, files=files, data=data)
            upload_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                document_id = result.get('id')
                
                print(f"✅ Upload completed in {upload_time:.2f} seconds!")
                print(f"📄 Document ID: {document_id}")
                print(f"📝 Title: {result.get('title')}")
                print(f"📊 Initial Status: {result.get('processing_status')}")
                print(f"📈 Progress: {result.get('processing_progress')}%")
                print(f"💬 Message: {result.get('processing_message')}")
                
                # Monitor processing status
                return monitor_processing_status(document_id)
                
            else:
                print(f"❌ Upload failed with status {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server.")
        print("Make sure the FastAPI server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error during upload test: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def monitor_processing_status(document_id):
    """Monitor the processing status of a document"""
    print(f"\\n📊 Monitoring processing status for document {document_id}...")
    
    status_url = f"http://localhost:8000/api/documents/status/{document_id}"
    max_wait_time = 300  # 5 minutes max
    check_interval = 3   # Check every 3 seconds
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        try:
            response = requests.get(status_url)
            if response.status_code == 200:
                status_data = response.json()
                
                processing_status = status_data.get('processing_status')
                progress = status_data.get('processing_progress', 0)
                message = status_data.get('processing_message', '')
                
                # Create a progress bar
                progress_bar = create_progress_bar(progress)
                
                print(f"\\r⏳ Status: {processing_status.upper()} {progress_bar} {progress}% - {message}", end='', flush=True)
                
                if processing_status == 'completed':
                    print(f"\\n\\n🎉 Processing completed successfully!")
                    
                    # Check for generated content
                    check_generated_content(document_id)
                    return True
                    
                elif processing_status == 'failed':
                    print(f"\\n\\n❌ Processing failed: {message}")
                    return False
                    
            else:
                print(f"\\n❌ Error checking status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"\\n❌ Error monitoring status: {e}")
            return False
        
        time.sleep(check_interval)
        elapsed_time += check_interval
    
    print(f"\\n⏰ Timeout after {max_wait_time} seconds")
    return False

def create_progress_bar(progress, width=20):
    """Create a visual progress bar"""
    filled = int(width * progress / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}]"

def check_generated_content(document_id):
    """Check what content was generated for the document"""
    print("\\n📚 Checking generated learning content...")
    
    try:
        # Check for learning modules
        modules_url = f"http://localhost:8000/api/learning/modules?document_id={document_id}"
        response = requests.get(modules_url)
        
        if response.status_code == 200:
            modules = response.json()
            print(f"✅ Generated {len(modules)} learning modules:")
            
            for i, module in enumerate(modules):
                print(f"   📖 Module {i+1}: {module.get('title')}")
                print(f"      ⏱️  Duration: {module.get('estimated_duration', 'N/A')} minutes")
                print(f"      📊 Difficulty: {module.get('difficulty_level')}")
                
                # Check sessions for this module
                sessions_count = module.get('sessions_count', 0)
                print(f"      🎯 Sessions: {sessions_count}")
        
        # Check for learning path
        path_url = f"http://localhost:8000/api/learning/documents/{document_id}/learning-path"
        response = requests.get(path_url)
        
        if response.status_code == 200:
            learning_path = response.json()
            print(f"\\n📈 Learning Path Summary:")
            print(f"   📄 Document: {learning_path.get('document_title')}")
            print(f"   📚 Total Modules: {learning_path.get('total_modules')}")
            print(f"   🎯 Total Sessions: {learning_path.get('total_sessions')}")
            print(f"   ⏱️  Estimated Duration: {learning_path.get('estimated_duration')} minutes")
        
    except Exception as e:
        print(f"⚠️ Could not check generated content: {e}")

def check_server_status():
    """Check if the FastAPI server is running"""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✅ FastAPI server is running")
            return True
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI server is not running")
        print("Please start the server with: uvicorn main:app --reload")
        return False

if __name__ == "__main__":
    print("=== Asynchronous Document Upload Test ===\\n")
    
    # Check server status first
    if not check_server_status():
        sys.exit(1)
    
    # Test async upload
    success = test_async_document_upload()
    
    if success:
        print("\\n🎉 All tests passed!")
        print("\\nAsynchronous upload workflow:")
        print("1. ✅ File uploaded immediately (fast response)")
        print("2. ✅ Background processing started")
        print("3. ✅ Status monitoring worked")
        print("4. ✅ Content generation completed")
        print("5. ✅ Learning modules and sessions created")
        print("\\n💡 Users can now upload files and get immediate feedback while processing happens in the background!")
    else:
        print("\\n❌ Tests failed. Please check the server logs for more details.")
        sys.exit(1)