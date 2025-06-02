# API Usage Examples

Practical examples for using the Dia FastAPI TTS Server API.

## Quick Start Examples

### Basic Text-to-Speech

```bash
# Simple generation
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "aria"}' \
  --output speech.wav

# With custom settings
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test with custom settings",
    "voice_id": "luna", 
    "speed": 1.2,
    "temperature": 0.8,
    "cfg_scale": 2.5
  }' \
  --output custom_speech.wav
```

### Async Generation

```bash
# Submit async job
JOB_ID=$(curl -s -X POST "http://localhost:7860/generate?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text for async processing...", "voice_id": "kai"}' \
  | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Check status
curl "http://localhost:7860/jobs/$JOB_ID"

# Download result when ready
curl "http://localhost:7860/jobs/$JOB_ID/result" --output async_result.wav
```

---

## Voice Management Examples

### Upload and Use Custom Voice

```bash
# 1. Upload audio prompt
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_custom_voice" \
  -F "audio_file=@my_voice_sample.wav"

# 2. Create voice mapping
curl -X POST "http://localhost:7860/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "custom_speaker",
    "style": "conversational",
    "primary_speaker": "S1",
    "audio_prompt": "my_custom_voice"
  }'

# 3. Generate with custom voice
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this uses my custom voice!", "voice_id": "custom_speaker"}' \
  --output custom_voice_output.wav
```

### List and Manage Voices

```bash
# List all available voices
curl "http://localhost:7860/voices" | jq '.voices'

# Get voice mappings
curl "http://localhost:7860/voice_mappings" | jq '.'

# Delete a custom voice
curl -X DELETE "http://localhost:7860/voice_mappings/custom_speaker"
curl -X DELETE "http://localhost:7860/audio_prompts/my_custom_voice"
```

---

## Advanced Usage Examples

### Bulk Generation Script

```bash
#!/bin/bash
# bulk_generate.sh - Generate multiple TTS files

TEXTS=(
    "Welcome to our service"
    "Thank you for your patience" 
    "Have a great day"
    "Please try again later"
)

VOICES=("aria" "atlas" "luna" "kai")

for i in "${!TEXTS[@]}"; do
    text="${TEXTS[$i]}"
    voice="${VOICES[$((i % ${#VOICES[@]}))]}"
    filename="bulk_${i}_${voice}.wav"
    
    echo "Generating: $text (Voice: $voice)"
    curl -X POST "http://localhost:7860/generate" \
      -H "Content-Type: application/json" \
      -d "{\"text\": \"$text\", \"voice_id\": \"$voice\"}" \
      --output "$filename"
    
    echo "Saved: $filename"
    sleep 1  # Rate limiting
done
```

### Voice Cloning Workflow

```bash
#!/bin/bash
# voice_cloning_workflow.sh - Complete voice cloning setup

PROMPT_ID="narrator_voice"
AUDIO_FILE="narrator_sample.wav"
VOICE_ID="professional_narrator"

echo "Step 1: Upload audio prompt..."
UPLOAD_RESULT=$(curl -s -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=$PROMPT_ID" \
  -F "audio_file=@$AUDIO_FILE")

echo "Upload result: $UPLOAD_RESULT"

echo "Step 2: Wait for transcription..."
sleep 5

echo "Step 3: Get transcript..."
METADATA=$(curl -s "http://localhost:7860/audio_prompts/metadata/$PROMPT_ID")
TRANSCRIPT=$(echo "$METADATA" | jq -r '.transcript')
echo "Auto-generated transcript: $TRANSCRIPT"

echo "Step 4: Create voice mapping..."
curl -s -X POST "http://localhost:7860/voice_mappings" \
  -H "Content-Type: application/json" \
  -d "{
    \"voice_id\": \"$VOICE_ID\",
    \"style\": \"professional\",
    \"primary_speaker\": \"S1\",
    \"audio_prompt\": \"$PROMPT_ID\",
    \"audio_prompt_transcript\": \"$TRANSCRIPT\"
  }"

echo "Step 5: Test voice generation..."
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This is a test of the cloned voice.\", \"voice_id\": \"$VOICE_ID\"}" \
  --output "test_${VOICE_ID}.wav"

echo "Voice cloning complete! Test file: test_${VOICE_ID}.wav"
```

---

## Programming Language Examples

### Python Examples

#### Simple Client Class

```python
import requests
import json
from pathlib import Path

class DiaTTSClient:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_speech(self, text, voice_id="aria", **kwargs):
        """Generate speech synchronously"""
        data = {"text": text, "voice_id": voice_id, **kwargs}
        response = self.session.post(f"{self.base_url}/generate", json=data)
        response.raise_for_status()
        return response.content
    
    def generate_speech_async(self, text, voice_id="aria", **kwargs):
        """Start async generation, return job ID"""
        data = {"text": text, "voice_id": voice_id, **kwargs}
        response = self.session.post(
            f"{self.base_url}/generate", 
            json=data, 
            params={"async_mode": "true"}
        )
        response.raise_for_status()
        return response.json()["job_id"]
    
    def get_job_status(self, job_id):
        """Check job status"""
        response = self.session.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_job_result(self, job_id):
        """Download job result"""
        response = self.session.get(f"{self.base_url}/jobs/{job_id}/result")
        response.raise_for_status()
        return response.content
    
    def upload_audio_prompt(self, prompt_id, audio_file_path):
        """Upload audio prompt for voice cloning"""
        files = {"audio_file": open(audio_file_path, "rb")}
        data = {"prompt_id": prompt_id}
        response = self.session.post(
            f"{self.base_url}/audio_prompts/upload",
            files=files,
            data=data
        )
        files["audio_file"].close()
        response.raise_for_status()
        return response.json()
    
    def list_voices(self):
        """Get available voices"""
        response = self.session.get(f"{self.base_url}/voices")
        response.raise_for_status()
        return response.json()["voices"]
    
    def health_check(self):
        """Check server health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Usage examples
if __name__ == "__main__":
    client = DiaTTSClient()
    
    # Basic generation
    audio = client.generate_speech("Hello, world!", voice_id="aria")
    with open("output.wav", "wb") as f:
        f.write(audio)
    
    # Async generation
    job_id = client.generate_speech_async("Long text for async processing...")
    print(f"Job started: {job_id}")
    
    # Wait for completion
    import time
    while True:
        status = client.get_job_status(job_id)
        print(f"Status: {status['status']}")
        if status["status"] == "completed":
            audio = client.get_job_result(job_id)
            with open("async_output.wav", "wb") as f:
                f.write(audio)
            break
        elif status["status"] == "failed":
            print("Job failed!")
            break
        time.sleep(1)
```

#### Batch Processing Example

```python
import concurrent.futures
import time
from pathlib import Path

def process_text_file(client, text_file, voice_id="aria"):
    """Process a text file and generate speech"""
    text = Path(text_file).read_text()
    
    # Split into chunks if too long
    max_length = 4000
    if len(text) > max_length:
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    else:
        chunks = [text]
    
    audio_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        audio = client.generate_speech(chunk, voice_id=voice_id)
        audio_chunks.append(audio)
    
    # Combine chunks (simple concatenation - for real use, consider audio processing)
    combined_audio = b''.join(audio_chunks)
    
    output_file = Path(text_file).stem + f"_{voice_id}.wav"
    Path(output_file).write_bytes(combined_audio)
    return output_file

# Batch process multiple files
client = DiaTTSClient()
text_files = ["story1.txt", "story2.txt", "story3.txt"]
voices = ["aria", "atlas", "luna"]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = []
    for text_file, voice in zip(text_files, voices):
        future = executor.submit(process_text_file, client, text_file, voice)
        futures.append(future)
    
    for future in concurrent.futures.as_completed(futures):
        output_file = future.result()
        print(f"Generated: {output_file}")
```

### JavaScript/Node.js Examples

```javascript
class DiaTTSClient {
    constructor(baseUrl = 'http://localhost:7860') {
        this.baseUrl = baseUrl;
    }
    
    async generateSpeech(text, voiceId = 'aria', options = {}) {
        const response = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice_id: voiceId, ...options })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        return response.arrayBuffer();
    }
    
    async generateSpeechAsync(text, voiceId = 'aria', options = {}) {
        const response = await fetch(`${this.baseUrl}/generate?async_mode=true`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice_id: voiceId, ...options })
        });
        
        const result = await response.json();
        return result.job_id;
    }
    
    async getJobStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}`);
        return response.json();
    }
    
    async getJobResult(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}/result`);
        return response.arrayBuffer();
    }
    
    async uploadAudioPrompt(promptId, audioFile) {
        const formData = new FormData();
        formData.append('prompt_id', promptId);
        formData.append('audio_file', audioFile);
        
        const response = await fetch(`${this.baseUrl}/audio_prompts/upload`, {
            method: 'POST',
            body: formData
        });
        
        return response.json();
    }
    
    async listVoices() {
        const response = await fetch(`${this.baseUrl}/voices`);
        const result = await response.json();
        return result.voices;
    }
}

// Usage examples
async function example() {
    const client = new DiaTTSClient();
    
    try {
        // Basic generation
        const audio = await client.generateSpeech("Hello, world!", "aria");
        
        // Save to file (Node.js)
        const fs = require('fs');
        fs.writeFileSync('output.wav', Buffer.from(audio));
        
        // Async generation
        const jobId = await client.generateSpeechAsync("Long text...");
        console.log(`Job started: ${jobId}`);
        
        // Poll for completion
        let status;
        do {
            await new Promise(resolve => setTimeout(resolve, 1000));
            status = await client.getJobStatus(jobId);
            console.log(`Status: ${status.status}`);
        } while (status.status === 'pending' || status.status === 'processing');
        
        if (status.status === 'completed') {
            const result = await client.getJobResult(jobId);
            fs.writeFileSync('async_output.wav', Buffer.from(result));
        }
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}
```

### Browser JavaScript Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dia TTS Web Interface</title>
</head>
<body>
    <div>
        <textarea id="textInput" placeholder="Enter text to convert to speech..."></textarea>
        <br>
        <select id="voiceSelect">
            <option value="aria">Aria</option>
            <option value="atlas">Atlas</option>
            <option value="luna">Luna</option>
            <option value="kai">Kai</option>
            <option value="zara">Zara</option>
            <option value="nova">Nova</option>
        </select>
        <br>
        <button onclick="generateSpeech()">Generate Speech</button>
        <button onclick="uploadAudioPrompt()">Upload Voice Sample</button>
        <br>
        <audio id="audioPlayer" controls style="display: none;"></audio>
        <div id="status"></div>
        <input type="file" id="audioFile" accept="audio/*" style="display: none;">
    </div>

    <script>
        const API_BASE = 'http://localhost:7860';
        
        async function generateSpeech() {
            const text = document.getElementById('textInput').value;
            const voice = document.getElementById('voiceSelect').value;
            const status = document.getElementById('status');
            
            if (!text.trim()) {
                alert('Please enter some text');
                return;
            }
            
            status.textContent = 'Generating speech...';
            
            try {
                const response = await fetch(`${API_BASE}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, voice_id: voice })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.src = audioUrl;
                audioPlayer.style.display = 'block';
                audioPlayer.play();
                
                status.textContent = 'Speech generated successfully!';
                
            } catch (error) {
                status.textContent = `Error: ${error.message}`;
            }
        }
        
        function uploadAudioPrompt() {
            document.getElementById('audioFile').click();
        }
        
        document.getElementById('audioFile').addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            
            const promptId = prompt('Enter a name for this voice:');
            if (!promptId) return;
            
            const status = document.getElementById('status');
            status.textContent = 'Uploading audio prompt...';
            
            try {
                const formData = new FormData();
                formData.append('prompt_id', promptId);
                formData.append('audio_file', file);
                
                const response = await fetch(`${API_BASE}/audio_prompts/upload`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                status.textContent = `Audio prompt uploaded: ${result.prompt_id}`;
                
                // Add to voice selector
                const voiceSelect = document.getElementById('voiceSelect');
                const option = document.createElement('option');
                option.value = promptId;
                option.textContent = promptId;
                voiceSelect.appendChild(option);
                
            } catch (error) {
                status.textContent = `Upload error: ${error.message}`;
            }
        });
        
        // Load available voices on page load
        async function loadVoices() {
            try {
                const response = await fetch(`${API_BASE}/voices`);
                const result = await response.json();
                
                const voiceSelect = document.getElementById('voiceSelect');
                voiceSelect.innerHTML = '';
                
                result.voices.forEach(voice => {
                    const option = document.createElement('option');
                    option.value = voice.voice_id;
                    option.textContent = voice.name;
                    voiceSelect.appendChild(option);
                });
                
            } catch (error) {
                console.error('Failed to load voices:', error);
            }
        }
        
        // Load voices when page loads
        loadVoices();
    </script>
</body>
</html>
```

---

## SillyTavern Integration Example

### SillyTavern Configuration

```json
{
  "provider": "Custom",
  "api_key": "dia-server",
  "url": "http://localhost:7860/v1/audio/speech",
  "voices": [
    "aria",
    "atlas", 
    "luna",
    "kai",
    "zara",
    "nova"
  ]
}
```

### Test SillyTavern Endpoint

```bash
# Test OpenAI-compatible endpoint
curl -X POST "http://localhost:7860/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dia",
    "input": "Hello from SillyTavern!",
    "voice": "luna",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output sillytavern_test.wav
```

---

## Monitoring and Debugging Examples

### Health Check Script

```bash
#!/bin/bash
# health_check.sh - Monitor server health

while true; do
    echo "=== $(date) ==="
    
    # Basic health check
    HEALTH=$(curl -s "http://localhost:7860/health" | jq -r '.status')
    echo "Server status: $HEALTH"
    
    # Get stats
    STATS=$(curl -s "http://localhost:7860/stats")
    echo "Active jobs: $(echo "$STATS" | jq -r '.requests.total')"
    echo "Success rate: $(echo "$STATS" | jq -r '.requests.success_rate')%"
    echo "Avg generation time: $(echo "$STATS" | jq -r '.performance.avg_generation_time')s"
    
    # GPU status (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU memory usage:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    fi
    
    echo ""
    sleep 30
done
```

### Performance Testing

```python
import time
import statistics
import concurrent.futures
from dia_client import DiaTTSClient  # Assuming the client class above

def performance_test():
    client = DiaTTSClient()
    
    test_texts = [
        "Short test.",
        "This is a medium length test sentence for performance evaluation.",
        "This is a much longer test sentence that contains significantly more text to evaluate how the TTS system performs with longer inputs and whether there are any performance degradations with increased text length."
    ]
    
    # Test each text length
    for i, text in enumerate(test_texts):
        print(f"\n=== Test {i+1}: {len(text)} characters ===")
        
        times = []
        for j in range(5):  # 5 runs each
            start_time = time.time()
            audio = client.generate_speech(text, voice_id="aria")
            end_time = time.time()
            
            generation_time = end_time - start_time
            times.append(generation_time)
            print(f"Run {j+1}: {generation_time:.2f}s")
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        
        print(f"Average: {avg_time:.2f}s ± {std_dev:.2f}s")
        print(f"Speed: {len(text) / avg_time:.1f} chars/second")
    
    # Concurrent test
    print(f"\n=== Concurrent Test ===")
    text = test_texts[1]  # Medium text
    
    def generate_concurrent():
        start_time = time.time()
        audio = client.generate_speech(text, voice_id="aria")
        return time.time() - start_time
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_concurrent) for _ in range(4)]
        times = [future.result() for future in futures]
    
    print(f"Concurrent average: {statistics.mean(times):.2f}s")
    print(f"Total time: {max(times):.2f}s")

if __name__ == "__main__":
    performance_test()
```

These examples should help you get started with integrating the Dia TTS API into your applications!