#!/usr/bin/env python3
"""
Script de configuración y demostración de WittgenLab.

Este script:
1. Verifica e instala las dependencias necesarias
2. Configura el entorno
3. Ejecuta una demostración completa
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


def print_header():
    """Imprime el header del script."""
    print("🧪 WITTGENLAB - Configuración y Demostración")
    print("=" * 60)
    print("Configurando el framework de evaluación de IA más completo...")
    print("=" * 60)


def check_python_version():
    """Verifica que la versión de Python sea compatible."""
    print("\n📋 Verificando versión de Python...")
    
    if sys.version_info < (3, 10):
        print("❌ Error: Se requiere Python 3.10 o superior")
        print(f"   Versión actual: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")


def check_poetry():
    """Verifica si Poetry está instalado."""
    print("\n📦 Verificando Poetry...")
    
    try:
        result = subprocess.run(['poetry', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️  Poetry no encontrado")
    print("   Instalando Poetry...")
    
    try:
        # Instalar Poetry usando el instalador oficial
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'poetry'
        ], check=True)
        print("✅ Poetry instalado correctamente")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error al instalar Poetry")
        return False


def install_dependencies():
    """Instala las dependencias del proyecto."""
    print("\n📚 Instalando dependencias...")
    
    try:
        # Instalar dependencias básicas con Poetry
        print("   Instalando dependencias básicas...")
        subprocess.run(['poetry', 'install'], check=True, cwd='.')
        
        # Instalar dependencias adicionales para la demo
        additional_deps = [
            'bert-score>=0.3.13',
            'langchain>=0.1.0',
            'langchain-openai>=0.1.0',
            'pydantic>=2.0.0'
        ]
        
        print("   Instalando dependencias adicionales...")
        for dep in additional_deps:
            subprocess.run([
                'poetry', 'add', dep
            ], check=True, cwd='.')
        
        print("✅ Dependencias instaladas correctamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al instalar dependencias: {e}")
        return False


def check_imports():
    """Verifica que las importaciones principales funcionen."""
    print("\n🔍 Verificando importaciones...")
    
    imports_to_check = [
        ('wittgenlab', 'EvalHub y configuración básica'),
        ('bert_score', 'BERTScore para evaluación semántica'),
        ('langchain', 'LangChain para LLM-as-a-Judge'),
        ('pydantic', 'Pydantic para validación de datos')
    ]
    
    all_good = True
    
    for module_name, description in imports_to_check:
        try:
            # Agregar src al path para wittgenlab
            if module_name == 'wittgenlab':
                sys.path.insert(0, str(Path('.') / 'src'))
            
            importlib.import_module(module_name)
            print(f"   ✅ {module_name:15} - {description}")
        except ImportError as e:
            print(f"   ❌ {module_name:15} - Error: {e}")
            all_good = False
    
    return all_good


def setup_api_keys():
    """Guía al usuario para configurar las API keys."""
    print("\n🔑 Configuración de API Keys")
    print("-" * 40)
    
    # Verificar OpenAI API Key
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY no configurada")
        print("   Para usar LLM-as-a-Judge necesitas configurar tu API key de OpenAI:")
        print("   1. Obtén tu API key en: https://platform.openai.com/api-keys")
        print("   2. Configúrala con: export OPENAI_API_KEY='tu-api-key-aquí'")
        print("   3. O agrégala al archivo .env en el directorio del proyecto")
        
        # Intentar cargar desde .env si existe
        env_file = Path('.env')
        if env_file.exists():
            print("   📄 Archivo .env encontrado - cargando variables...")
            with open(env_file) as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        key = line.split('=', 1)[1].strip().strip('"\'')
                        os.environ['OPENAI_API_KEY'] = key
                        print("   ✅ OPENAI_API_KEY cargada desde .env")
                        break
        
        return False
    else:
        print("✅ OPENAI_API_KEY configurada")
        return True


def run_basic_test():
    """Ejecuta un test básico del framework."""
    print("\n🧪 Ejecutando test básico...")
    
    try:
        # Importar y probar funcionalidad básica
        sys.path.insert(0, str(Path('.') / 'src'))
        from wittgenlab import EvalHub
        
        evaluator = EvalHub()
        
        # Test básico de métricas
        predictions = ["El gato está en el tejado."]
        references = ["Un gato se encuentra sobre el techo."]
        
        # Probar métricas básicas
        try:
            results = evaluator.evaluate(
                predictions=predictions,
                references=references,
                metrics=['bleu', 'rouge'],
                task='test'
            )
            print("   ✅ Métricas básicas funcionando")
        except Exception as e:
            print(f"   ⚠️  Métricas básicas: {e}")
        
        # Probar BERTScore si está disponible
        try:
            bertscore_results = evaluator.evaluate(
                predictions=predictions,
                references=references,
                metrics=['bertscore'],
                task='test'
            )
            print("   ✅ BERTScore funcionando")
        except Exception as e:
            print(f"   ⚠️  BERTScore: {e}")
        
        # Probar LLM Judge si API key está disponible
        if os.environ.get('OPENAI_API_KEY'):
            try:
                from wittgenlab.judges import create_judge
                judge = create_judge(model_name="gpt-4o-mini")
                print("   ✅ LLM Judge configurado")
            except Exception as e:
                print(f"   ⚠️  LLM Judge: {e}")
        
        print("✅ Test básico completado")
        return True
        
    except Exception as e:
        print(f"❌ Error en test básico: {e}")
        return False


def run_complete_demo():
    """Ejecuta la demostración completa."""
    print("\n🚀 Ejecutando demostración completa...")
    print("   Esto puede tomar unos minutos...")
    
    try:
        # Ejecutar el script de demostración
        demo_script = Path('examples') / 'complete_demo.py'
        
        if demo_script.exists():
            subprocess.run([sys.executable, str(demo_script)], check=True)
            print("✅ Demostración completa ejecutada")
            return True
        else:
            print("❌ Script de demostración no encontrado")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar demostración: {e}")
        return False


def print_next_steps():
    """Imprime los próximos pasos para el usuario."""
    print("\n🎯 PRÓXIMOS PASOS")
    print("=" * 60)
    print("1. 📖 Lee la documentación en README.md")
    print("2. 🔧 Configura tu OPENAI_API_KEY para usar LLM-as-a-Judge")
    print("3. 🎮 Ejecuta los ejemplos:")
    print("   • python examples/basic_usage.py")
    print("   • python examples/advanced_usage.py")
    print("   • python examples/complete_demo.py")
    print("4. 🚀 Adapta los ejemplos a tus propios datos")
    print("5. 📊 Explora métricas adicionales y benchmarks")
    print("")
    print("💡 CONSEJOS:")
    print("   • Usa 'poetry shell' para activar el entorno virtual")
    print("   • Configura el archivo .env con tus API keys")
    print("   • Revisa examples/ para más casos de uso")
    print("")
    print("🆘 AYUDA:")
    print("   • GitHub: https://github.com/Robert-Gomez-AI/wittgenlab")
    print("   • Email: robertgomez.datascience@gmail.com")


def main():
    """Función principal del script de configuración."""
    print_header()
    
    # Paso 1: Verificar Python
    check_python_version()
    
    # Paso 2: Verificar/Instalar Poetry
    if not check_poetry():
        print("❌ No se pudo configurar Poetry. Saliendo...")
        sys.exit(1)
    
    # Paso 3: Instalar dependencias
    if not install_dependencies():
        print("❌ No se pudieron instalar las dependencias. Saliendo...")
        sys.exit(1)
    
    # Paso 4: Verificar importaciones
    if not check_imports():
        print("⚠️  Algunas importaciones fallaron, pero continuando...")
    
    # Paso 5: Configurar API keys
    api_keys_ok = setup_api_keys()
    
    # Paso 6: Test básico
    if not run_basic_test():
        print("⚠️  Test básico falló, pero la instalación puede estar bien")
    
    # Paso 7: Demostración completa (opcional)
    print("\n❓ ¿Ejecutar demostración completa? (s/N): ", end="")
    response = input().lower().strip()
    
    if response in ['s', 'si', 'y', 'yes']:
        run_complete_demo()
    else:
        print("   Saltando demostración completa")
    
    # Paso 8: Próximos pasos
    print_next_steps()
    
    print("\n🎉 ¡Configuración completada!")
    print("WittgenLab está listo para usar. ¡Disfruta evaluando tus modelos!")


if __name__ == "__main__":
    main() 