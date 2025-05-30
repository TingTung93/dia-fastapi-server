#!/usr/bin/env python3
"""Benchmark different audio prompt approaches on GPU"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import time
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, 'src')

class AudioPromptBenchmark:
    def __init__(self):
        self.model = None
        self.device = None
        self.results = []
        
    def setup_gpu(self):
        """Setup GPU and load model"""
        print("🚀 GPU Setup\n")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ Memory: {gpu_memory:.1f} GB")
        print(f"✅ Compute Capability: {torch.cuda.get_device_capability()}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return True
    
    def load_model(self):
        """Load Dia model on GPU"""
        print("\n🔄 Loading Dia model...")
        
        try:
            from dia import Dia
        except ImportError:
            raise ImportError("Dia not found. Activate conda environment first.")
        
        # Use optimal precision
        if torch.cuda.is_bf16_supported():
            compute_dtype = "bfloat16"
        else:
            compute_dtype = "float16"
        
        print(f"✅ Using {compute_dtype} precision")
        
        # Load with progress tracking
        start = time.time()
        self.model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B",
            compute_dtype=compute_dtype,
            device=self.device
        )
        load_time = time.time() - start
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Warm up GPU
        print("🔥 Warming up GPU...")
        self.model.generate("[S1] Test [S1]", verbose=False)
        torch.cuda.synchronize()
        
        return True
    
    def load_audio_prompt(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load and analyze audio prompt"""
        if not os.path.exists(filepath):
            return None
        
        data, sr = sf.read(filepath)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        return {
            "path": filepath,
            "data": data,
            "sample_rate": sr,
            "duration": len(data) / sr,
            "samples": len(data)
        }
    
    def benchmark_generation(self, name: str, text: str, 
                           audio_prompt: Optional[str] = None,
                           audio_data: Optional[np.ndarray] = None,
                           **kwargs):
        """Benchmark a single generation"""
        print(f"\n🧪 {name}")
        
        # Prepare parameters
        params = {
            "temperature": kwargs.get("temperature", 1.0),
            "cfg_scale": kwargs.get("cfg_scale", 3.0),
            "top_p": kwargs.get("top_p", 0.95),
            "use_torch_compile": kwargs.get("use_torch_compile", False),
            "verbose": False
        }
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Measure GPU memory before
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        
        try:
            # Time generation
            torch.cuda.synchronize()
            start = time.time()
            
            # Try different audio prompt formats
            if audio_prompt:
                # Method 1: File path
                audio = self.model.generate(text, audio_prompt=audio_prompt, **params)
            elif audio_data is not None:
                # Method 2: Audio data (if supported)
                try:
                    audio = self.model.generate(text, audio_prompt=audio_data, **params)
                except:
                    # Fallback to no audio prompt
                    audio = self.model.generate(text, **params)
            else:
                # No audio prompt
                audio = self.model.generate(text, **params)
            
            torch.cuda.synchronize()
            gen_time = time.time() - start
            
            # Measure GPU memory after
            mem_after = torch.cuda.memory_allocated()
            mem_used = (mem_after - mem_before) / 1024**3
            
            if audio is not None and len(audio) > 0:
                duration = len(audio) / 44100
                realtime_factor = duration / gen_time
                
                # Save output
                output_file = f"benchmark_{name}.wav"
                sf.write(output_file, audio, 44100)
                
                result = {
                    "name": name,
                    "success": True,
                    "gen_time": gen_time,
                    "audio_duration": duration,
                    "realtime_factor": realtime_factor,
                    "tokens_per_sec": (duration * 86) / gen_time,  # ~86 tokens/sec
                    "gpu_memory_gb": mem_used,
                    "output_file": output_file
                }
                
                print(f"✅ Generated {duration:.2f}s in {gen_time:.2f}s")
                print(f"   Speed: {realtime_factor:.1f}x realtime")
                print(f"   Memory: +{mem_used:.2f}GB")
            else:
                result = {
                    "name": name,
                    "success": False,
                    "error": "No audio generated"
                }
                print("❌ No audio generated")
                
        except Exception as e:
            result = {
                "name": name,
                "success": False,
                "error": str(e)
            }
            print(f"❌ Error: {e}")
        
        self.results.append(result)
        return result
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "="*60)
        print("🏁 Running Benchmarks\n")
        
        # Test text
        test_text = "Hello, this is a comprehensive test of the voice cloning system. I should sound natural and clear."
        
        # Load audio prompt if available
        audio_prompt_path = "audio_prompts/seraphina_voice.wav"
        audio_info = self.load_audio_prompt(audio_prompt_path)
        
        if audio_info:
            print(f"✅ Loaded audio prompt: {audio_info['duration']:.1f}s @ {audio_info['sample_rate']}Hz")
        
        # Benchmark configurations
        configs = [
            # Baseline tests
            ("baseline_s1", "[S1] " + test_text + " [S1]", None, None, {}),
            ("baseline_s2", "[S2] " + test_text + " [S2]", None, None, {}),
            
            # With torch compile (if supported)
            ("s2_compiled", "[S2] " + test_text + " [S2]", None, None, 
             {"use_torch_compile": torch.cuda.get_device_capability()[0] >= 8}),
        ]
        
        # Add audio prompt tests if available
        if audio_info:
            configs.extend([
                # Audio prompt as file path
                ("prompt_path_s1", "[S1] " + test_text + " [S1]", 
                 audio_info['path'], None, {}),
                ("prompt_path_s2", "[S2] " + test_text + " [S2]", 
                 audio_info['path'], None, {}),
                
                # With transcript prepended
                ("prompt_with_transcript", 
                 "This is Seraphina speaking. [S2] " + test_text + " [S2]",
                 audio_info['path'], None, {}),
                
                # Different CFG scales
                ("prompt_cfg_2", "[S2] " + test_text + " [S2]",
                 audio_info['path'], None, {"cfg_scale": 2.0}),
                ("prompt_cfg_5", "[S2] " + test_text + " [S2]",
                 audio_info['path'], None, {"cfg_scale": 5.0}),
                
                # Audio data instead of path (experimental)
                ("prompt_data_s2", "[S2] " + test_text + " [S2]",
                 None, audio_info['data'], {}),
            ])
        
        # Run benchmarks
        for config in configs:
            name, text, audio_path, audio_data, params = config
            self.benchmark_generation(name, text, audio_path, audio_data, **params)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("📊 Benchmark Summary\n")
        
        successful = [r for r in self.results if r['success']]
        
        if successful:
            print("✅ Successful generations:\n")
            print(f"{'Name':<25} {'Time (s)':<10} {'RT Factor':<10} {'Tokens/s':<10} {'GPU (GB)':<10}")
            print("-" * 75)
            
            for r in successful:
                print(f"{r['name']:<25} {r['gen_time']:<10.2f} "
                      f"{r['realtime_factor']:<10.1f} {r['tokens_per_sec']:<10.1f} "
                      f"{r['gpu_memory_gb']:<10.2f}")
            
            # Find fastest
            fastest = min(successful, key=lambda x: x['gen_time'])
            print(f"\n⚡ Fastest: {fastest['name']} ({fastest['gen_time']:.2f}s)")
            
            # Compare with/without audio prompt
            baseline = next((r for r in successful if 'baseline' in r['name']), None)
            with_prompt = next((r for r in successful if 'prompt' in r['name'] and 'baseline' not in r['name']), None)
            
            if baseline and with_prompt:
                diff = with_prompt['gen_time'] - baseline['gen_time']
                print(f"\n📊 Audio prompt overhead: +{diff:.2f}s ({diff/baseline['gen_time']*100:.1f}%)")
        
        failed = [r for r in self.results if not r['success']]
        if failed:
            print(f"\n❌ Failed generations: {len(failed)}")
            for r in failed:
                print(f"   - {r['name']}: {r['error']}")
        
        print("\n💡 Analysis:")
        print("1. Compare audio files to check if audio prompt is working")
        print("2. If all sound the same, audio prompt may not be loading")
        print("3. Check cfg_scale variations for prompt adherence")
        print("4. S1 vs S2 should show clear gender differences")

def main():
    print("🚀 GPU-Accelerated Audio Prompt Benchmark\n")
    
    benchmark = AudioPromptBenchmark()
    
    try:
        # Setup
        benchmark.setup_gpu()
        benchmark.load_model()
        
        # Run benchmarks
        benchmark.run_benchmarks()
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return
    
    print("\n✅ Benchmark complete!")
    print("   Check the generated benchmark_*.wav files")
    print("   Use an audio editor to A/B compare them")

if __name__ == "__main__":
    main()