# Instrucciones para subir a GitHub

El proyecto está inicializado con git y listo para compartir. Aquí está cómo hacerlo:

## Paso 1: Crear un repositorio en GitHub

1. Ve a https://github.com/new
2. Crea un repositorio nuevo con nombre: `cowtrack` (o el que prefieras)
3. **NO** inicialices con README, .gitignore, ni licencia (ya los tenemos localmente)
4. Copia la URL del repositorio (será algo como `https://github.com/tu-usuario/cowtrack.git`)

## Paso 2: Hacer push a GitHub

Desde PowerShell en la carpeta del proyecto:

```bash
# Agregar el remote de GitHub
git remote add origin https://github.com/TU-USUARIO/cowtrack.git

# Renombrar rama a 'main' (GitHub usa esto por defecto)
git branch -M main

# Hacer push del código
git push -u origin main
```

## Paso 3: Invitar colaboradores

1. Ve a Settings → Collaborators en tu repositorio GitHub
2. Haz clic en "Add people"
3. Invita a tus compañeros con su usuario de GitHub

## Paso 4: Compartir el link

Comparte con tu equipo:
```
https://github.com/TU-USUARIO/cowtrack
```

---

## Para tus compañeros: cómo clonar el proyecto

```bash
# Clonar el repositorio
git clone https://github.com/TU-USUARIO/cowtrack.git
cd cowtrack

# Crear virtual environment
python -m venv .venv
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ver documentación
# Leer reports/ENTREGABLES_estado_actual.md
```

---

## Estructura compartida

```
GitHub
 ↳ Code (scripts, src, configs)
 ↳ Reports (documentación, visuales, análisis)
 ↳ Issues (para tareas y bugs)
 ↳ Discussions (para conversaciones)
 ↳ .gitignore (excluye datos pesados y checkpoints)

NOTA: data/raw/ y outputs/checkpoints/ no se suben
      (archivo > 100MB son excluidos por .gitignore)
```

---

## Qué está compartido en GitHub

✅ Código fuente (scripts, src/)
✅ Documentación (reports *.md)
✅ Configuración (requirements.txt, configs/)
✅ Tests (tests/)
✅ Visuales pequeñas (reports/*).md)

❌ Datasets grandes (data/raw/** - potencialmente GB)
❌ Checkpoints (*.pt - 100+ MB)
❌ Índices FAISS (*.index - 100+ MB)
❌ Archivos temporales

---

## Flujo de trabajo recomendado

Para ti y tus compañeros:

```
1. Cambios locales en rama propia
   git checkout -b tu-nombre/feature-description

2. Haz commit de tus cambios
   git add <archivo>
   git commit -m "Descripción breve"

3. Push a GitHub
   git push origin tu-nombre/feature-description

4. Abre Pull Request en GitHub

5. Revisa, merged, y actualiza localmente
   git pull origin main
```

---

## Instrucciones rápidas de push

**Primera vez (una sola vez):**
```bash
git remote add origin https://github.com/TU-USUARIO/cowtrack.git
git branch -M main
git push -u origin main
```

**Próximas veces (más rápido):**
```bash
git push
```

---

## ¿Necesitas ayuda?

- Documentación del proyecto: `reports/ENTREGABLES_estado_actual.md`
- Estado actual: `reports/reid/what_works_today.md`
- Timeline: `reports/dataset_usage_plan.md`

---

**Listo para compartir! 🚀**
