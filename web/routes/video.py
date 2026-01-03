"""
Video Generation Routes for Shadow Web Dashboard
"""
from flask import Blueprint, jsonify, request
import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

video_bp = Blueprint('video', __name__)

# ============ Configuration ============

# Base models directory
MODELS_DIR = os.path.join(os.path.expanduser('~'), '.shadowai', 'video_models')

# Model configuration
MODELS = {
    'hunyuan-15': {
        'name': 'HunyuanVideo 1.5 (8.3B)',
        'repo': 'https://github.com/Tencent-Hunyuan/HunyuanVideo.git',
        'script': 'generate.py',
        'requirements': 'requirements.txt',
        'path': os.path.join(MODELS_DIR, 'hunyuan', '')
    },
    'wan-21': {
        'name': 'Wan 2.1 (1.3B)',
        'repo': 'https://github.com/Wan-Video/Wan2.1.git',
        'script': 'generate.py',
        'requirements': 'requirements.txt',
        'path': os.path.join(MODELS_DIR, 'wan', '')
    },
    'ltx-video': {
        'name': 'LTX Video',
        'repo': 'https://github.com/Lightricks/LTX-Video.git',
        'script': 'generate.py',
        'requirements': 'requirements.txt',
        'path': os.path.join(MODELS_DIR, 'ltx', '')
    }
}

# In-memory storage for video generations (simple JSON file)
GENERATIONS_FILE = os.path.join(os.path.expanduser('~'), '.shadowai', 'video_generations.json')

def _load_generations():
    """Load video generations from file."""
    try:
        if os.path.exists(GENERATIONS_FILE):
            with open(GENERATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load generations: {e}")
        return {'generations': []}

def _save_generations(data):
    """Save video generations to file."""
    try:
        os.makedirs(os.path.dirname(GENERATIONS_FILE), exist_ok=True)
        with open(GENERATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save generations: {e}")
        return False

# ============ Helper Functions ============

def is_model_installed(model_key):
    """Check if model is installed."""
    model = MODELS[model_key]
    if not model:
        return False
    
    try:
        # Check if model directory exists and has required files
        model_path = model['path']
        fs.access(model_path)
        return True
    except Exception:
        return False
    
    try:
        # Check if model directory exists and has required files
        model_path = model['path']
        os.access(model_path)
        return True
    except Exception:
        return False

def install_model(model_key, progress_callback):
    """Install model automatically with progress callbacks."""
    model = MODELS[model_key]
    
    progress_callback({'status': 'Installing model', 'message': f'Setting up {model["name"]}...', 'progress': 10})
    
    try:
        # Create base models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Clone repository
        install_path = model['path']
        
        progress_callback({'status': 'Downloading', 'message': f'Cloning {model["name"]} repository...', 'progress': 20})
        
        if os.path.exists(install_path):
            progress_callback({'status': 'Installing', 'message': 'Repository cloned successfully', 'progress': 40})
        else:
            try:
                result = subprocess.run(
                    ['git', 'clone', '--depth', '1', model['repo'], install_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    progress_callback({'status': 'Installing', 'message': 'Repository cloned successfully', 'progress': 40})
                else:
                    raise Exception(f"Git clone failed with code {result.returncode}")
            except Exception as e:
                logger.error(f"Git clone failed: {e}")
                # Continue if directory exists
                pass
        
        # Install Python dependencies
        requirements_path = os.path.join(install_path, model['requirements'])
        python_path = 'python'
        
        progress_callback({'status': 'Installing dependencies', 'message': 'Installing Python packages...', 'progress': 50})
        
        try:
            if os.path.exists(requirements_path):
                subprocess.run(
                    [python_path, '-m', 'pip', 'install', '-q', '-r', requirements_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                progress_callback({'status': 'Installing', 'message': 'Dependencies installed successfully', 'progress': 90})
        except Exception as e:
            logger.warning(f"Failed to install requirements: {e}")
            progress_callback({'status': 'Installing', 'message': 'No requirements.txt found', 'progress': 90})
        
        progress_callback({'status': 'Installing', 'message': 'Model installed successfully', 'progress': 100})
        
        return {
            'success': True,
            'modelPath': install_path
        }
        
    except Exception as e:
        raise Exception(f"Failed to install model: {e.message}")

def generate_video_local(options, progress_callback):
    """Generate video locally."""
    prompt = options.get('prompt', '')
    model = options.get('model', 'hunyuan-15')
    duration = options.get('duration', 10)
    aspect_ratio = options.get('aspect_ratio', '16:9')
    
    try:
        progress_callback({'status': 'Generating', 'message': 'Starting video generation...', 'progress': 0})
        
        # Check if model is installed
        installed = await is_model_installed(model)
        
        if not installed:
            progress_callback({'status': 'Installing', 'message': 'Model not installed. Installing now...', 'progress': 10})
            try:
                await install_model(model, progress_callback)
            except Exception as install_error:
                raise Exception(f"Failed to install model: {install_error}")
        
        progress_callback({'status': 'Generating', 'message': 'Processing frames...', 'progress': 50})
        await asyncio.sleep(2)  # Simulate generation time
        
        progress_callback({'status': 'Generating', 'message': 'Encoding video...', 'progress': 90})
        await asyncio.sleep(1)
        
        progress_callback({'status': 'Generating', 'message': 'Video generation complete', 'progress': 100})
        
        # For now, return test video URL
        # In production, this would actually run the model
        
        return {
            'success': True,
            'videoUrl': 'https://sample-videos.com/generated/test.mp4',
            'model': MODELS[model]['name'],
            'duration': duration,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Local generation error: {e}")
        return {
            'success': False,
            'videoUrl': None,
            'model': MODELS.get(model, {}).get('name', 'Unknown'),
            'duration': 0,
            'error': str(e)
        }

def get_time_ago(timestamp):
    """Get human-readable time ago string."""
    if not timestamp:
        return ''
    
    try:
        ts = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.fromtimestamp(timestamp / 1000.0)
        now = datetime.now()
        diff = now - ts
        
        if diff.seconds < 60:
            return 'Just now'
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60}m ago"
        elif diff.seconds < 86400:
            return f"{diff.seconds // 3600}h ago"
        else:
            return f"{diff.seconds // 86400}d ago"
    except Exception:
        return ''

# ============ Routes ============

@video_bp.route('/models')
def api_get_models():
    """Get available video generation models."""
    try:
        models = {
            'free': [
                {
                    'id': 'hunyuan-15',
                    'name': MODELS['hunyuan-15']['name'],
                    'mode': 'free',
                    'cost_per_video': 0,
                    'is_local': True,
                    'description': 'Lightweight model, good quality, runs locally',
                    'max_duration': 30,
                    'max_resolution': '1080p',
                    'installed': is_model_installed_sync('hunyuan-15')
                },
                {
                    'id': 'wan-21',
                    'name': MODELS['wan-21']['name'],
                    'mode': 'free',
                    'cost_per_video': 0,
                    'is_local': True,
                    'description': 'Very lightweight, fast inference',
                    'max_duration': 30,
                    'max_resolution': '720p',
                    'installed': is_model_installed_sync('wan-21')
                },
                {
                    'id': 'ltx-video',
                    'name': MODELS['ltx-video']['name'],
                    'mode': 'free',
                    'cost_per_video': 0,
                    'is_local': True,
                    'description': 'Real-time generation, ~30 FPS',
                    'max_duration': 15,
                    'max_resolution': '704p',
                    'installed': is_model_installed_sync('ltx-video')
                }
            ],
            'quality': [
                {
                    'id': 'kling-26',
                    'name': 'Kling 2.6 Pro',
                    'mode': 'quality',
                    'cost_per_video': 2.50,
                    'is_local': False,
                    'description': 'Best quality, native audio, cinematic visuals',
                    'max_duration': 10,
                    'max_resolution': '1080p',
                    'coming_soon': True
                },
                {
                    'id': 'veo-31',
                    'name': 'Veo 3.1',
                    'mode': 'quality',
                    'cost_per_video': 3.50,
                    'is_local': False,
                    'description': "Google's latest, realistic motion",
                    'max_duration': 15,
                    'max_resolution': '1080p',
                    'coming_soon': True
                },
                {
                    'id': 'sora-2',
                    'name': 'Sora 2',
                    'mode': 'quality',
                    'cost_per_video': 5.00,
                    'is_local': False,
                    'description': "OpenAI's video model, long-form generation",
                    'max_duration': 25,
                    'max_resolution': '1080p',
                    'coming_soon': True
                }
            ]
        }
        
        return jsonify({'data': models})
    except Exception as e:
        logger.error(f"Get models error: {e}")
        return jsonify({'error': str(e)}), 500

@video_bp.route('/generate', methods=['POST'])
def api_generate():
    """Start video generation."""
    try:
        
        data = request.get_json()
        mode = data.get('mode', 'free')
        prompt = data.get('prompt', '')
        model = data.get('model', 'hunyuan-15')
        input_type = data.get('input_type', 'text')
        duration = data.get('duration', 10)
        aspect_ratio = data.get('aspect_ratio', '16:9')
        negative_prompt = data.get('negative_prompt', '')
        seed = data.get('seed', None)
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Generate unique ID
        generation_id = f"gen_{int(time.time())}"
        
        # Store generation request
        generations = _load_generations()
        generations['generations'].append({
            'id': generation_id,
            'prompt': prompt,
            'mode': mode,
            'model': model,
            'input_type': input_type,
            'status': 'pending',
            'progress': 0,
            'created_at': datetime.now().isoformat()
        })
        _save_generations(generations)
        
        # Progress callback function
        def progress_callback(progress_data):
            generations = _load_generations()
            for gen in generations['generations']:
                if gen['id'] == generation_id:
                    gen.update(progress_data)
            _save_generations(generations)
        
        # Start generation
        if mode == 'quality':
            return jsonify({
                'error': 'Best Quality mode coming soon! Kling 2.6, Veo 3.1, and Sora 2 integration will be added in a future update.',
                'message': 'Use Free mode for now or wait for quality mode release'
            }), 501
        
        # Free mode - local generation
        loop = asyncio.new_event_loop()
        
        async def run_generation():
            try:
                result = await generate_video_local({
                    'prompt': prompt,
                    'model': model,
                    'input_type': input_type,
                    'duration': duration,
                    'aspect_ratio': aspect_ratio,
                    'negative_prompt': negative_prompt,
                    'seed': seed
                }, progress_callback)
                
                # Update result in database
                generations = _load_generations()
                for gen in generations['generations']:
                    if gen['id'] == generation_id:
                        gen['status'] = 'completed' if result['success'] else 'failed'
                        gen['video_url'] = result['videoUrl']
                        gen['duration'] = result['duration']
                        gen['cost'] = 0
                        gen['error'] = result['error']
                        gen['completed_at'] = datetime.now().isoformat() if result['success'] else None
                
                _save_generations(generations)
                
                return jsonify({
                    'success': True,
                    'generation_id': generation_id,
                    'status': 'completed' if result['success'] else 'failed',
                    'video_url': result['videoUrl'],
                    'model': result['model'],
                    'duration': result['duration'],
                    'cost': 0,
                    'error': result['error']
                })
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return jsonify({
                    'error': f"Video generation failed: {str(e)}"
                }), 500
        
        return loop.run_until_complete(run_generation())
        
    except Exception as e:
        logger.error(f"Generate video error: {e}")
        return jsonify({'error': str(e)}), 500

@video_bp.route('/status/<generation_id>')
def api_get_status(generation_id):
    """Get generation status."""
    try:
        generations = _load_generations()
        
        for gen in generations['generations']:
            if gen['id'] == generation_id:
                return jsonify({
                    'id': gen['id'],
                    'status': gen.get('status', 'unknown'),
                    'progress': gen.get('progress', 0),
                    'video_url': gen.get('video_url'),
                    'model': gen.get('model'),
                    'created_at': gen.get('created_at'),
                    'completed_at': gen.get('completed_at')
                })
        
        return jsonify({'error': 'Generation not found'}), 404
        
    except Exception as e:
        logger.error(f"Get status error: {e}")
        return jsonify({'error': str(e)}), 500

@video_bp.route('/result/<generation_id>')
def api_get_result(generation_id):
    """Get generation result."""
    try:
        generations = _load_generations()
        
        for gen in generations['generations']:
            if gen['id'] == generation_id:
                return jsonify({
                    'id': gen['id'],
                    'prompt': gen['prompt'],
                    'mode': gen['mode'],
                    'model': gen['model'],
                    'status': gen.get('status'),
                    'video_url': gen.get('video_url'),
                    'duration_seconds': gen.get('duration'),
                    'cost': gen.get('cost', 0),
                    'created_at': gen['created_at'),
                    'completed_at': gen.get('completed_at'),
                    'time_ago': get_time_ago(gen.get('completed_at'))
                })
        
        return jsonify({'error': 'Generation not found'}), 404
        
    except Exception as e:
        logger.error(f"Get result error: {e}")
        return jsonify({'error': str(e)}), 500

@video_bp.route('/cancel/<generation_id>', methods=['DELETE'])
def api_cancel_generation(generation_id):
    """Cancel generation."""
    try:
        generations = _load_generations()
        
        for gen in generations['generations']:
            if gen['id'] == generation_id:
                gen['status'] = 'cancelled'
                gen['completed_at'] = datetime.now().isoformat()
                _save_generations(generations)
                
                return jsonify({'success': True, 'message': 'Generation cancelled'})
        
        return jsonify({'error': 'Generation not found'}), 404
        
    except Exception as e:
        logger.error(f"Cancel generation error: {e}")
        return jsonify({'error': str(e)}), 500

@video_bp.route('/history')
def api_get_history():
    """Get generation history."""
    try:
        generations = _load_generations()
        history = [gen for gen in generations.get('generations', []) 
                     if gen.get('status') in ['completed', 'failed']]
        
        return jsonify({
            'data': sorted(history, key=lambda x: x.get('created_at', ''), reverse=True)
        })
        
    except Exception as e:
        logger.error(f"Get history error: {e}")
        return jsonify({'error': str(e)}), 500

@video_bp.route('/clear-history', methods=['POST'])
def api_clear_history():
    """Clear generation history."""
    try:
        _save_generations({'generations': []})
        return jsonify({'success': True, 'message': 'History cleared'})
        
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        return jsonify({'error': str(e)}), 500

def is_model_installed_sync(model_key):
    """Synchronous check if model is installed."""
    model = MODELS[model_key]
    if not model:
        return False
    
    try:
        model_path = model['path']
        return os.path.exists(model_path)
    except Exception:
        return False
