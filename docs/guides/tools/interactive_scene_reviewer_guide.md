# Interactive Scene Cluster Reviewer Guide

## Overview

The Interactive Scene Cluster Reviewer is a web-based tool for reviewing and curating scene clustering results from the background segmentation and clustering pipeline. It provides a visual interface for:

- **Viewing cluster thumbnails** in a grid layout
- **Merging clusters** via drag & drop
- **Moving individual images** between clusters
- **Renaming clusters** with semantic labels (e.g., "indoor_home", "outdoor_plaza")
- **Deleting noise clusters**
- **Exporting reviewed results** back to the pipeline

## Features

### Grid View
- Visual overview of all scene clusters
- Thumbnail preview of first 12 images per cluster
- Cluster statistics (size, count)
- Sort by size or name

### Cluster Operations
- **Rename**: Click the pencil icon (✏️) to rename a cluster with meaningful labels
- **Delete**: Click the trash icon (🗑️) to remove noise or unwanted clusters
- **Merge**: Drag and drop images between clusters to merge similar scenes
- **View Details**: Click on cluster name to see all images in detail

### Keyboard Shortcuts
- `G` - Return to grid view
- `E` - Export reviewed clusters

## Usage

### Basic Command Line

```bash
python scripts/generic/clustering/interactive_scene_reviewer.py \
  --cluster-dir /path/to/scene_clusters_giant \
  --output-dir /path/to/scene_clusters_reviewed \
  --port 5001
```

### Using Batch Script

For convenience, use the provided batch script:

```bash
# Review Turning Red scenes
bash scripts/batch/review_scene_clusters.sh turning-red 5001

# Review other movies
bash scripts/batch/review_scene_clusters.sh luca 5002
bash scripts/batch/review_scene_clusters.sh up 5003
bash scripts/batch/review_scene_clusters.sh onward 5004
```

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--cluster-dir` | Yes | Directory containing scene clustering results (scene_* folders) |
| `--output-dir` | Yes | Output directory for reviewed clusters |
| `--port` | No | Server port (default: 5001) |
| `--no-browser` | No | Don't open browser automatically |

## Workflow

### 1. Launch the Reviewer

```bash
bash scripts/batch/review_scene_clusters.sh turning-red 5001
```

The web interface will automatically open in your browser at `http://localhost:5001`

### 2. Review Clusters

**Grid View** shows all clusters with:
- Cluster name (default: scene_000, scene_001, etc.)
- Image count badge
- Thumbnail grid (first 12 images)

**Actions you can take:**
- **Sort clusters**: Use "Sort by Size" or "Sort by Name" buttons
- **View details**: Click on cluster name to see all images
- **Rename**: Click ✏️ icon and enter semantic name (e.g., "miguel_house_kitchen", "santa_cecilia_plaza")
- **Delete**: Click 🗑️ icon to remove unwanted clusters

### 3. Merge Similar Clusters

**Drag & Drop Method:**
1. Click and drag an image from one cluster
2. Drop it onto another cluster card
3. The image moves to the target cluster
4. Both clusters are marked as "modified"

**Use Cases:**
- Merge duplicate scenes (same location, different lighting)
- Combine scene variations (day/night versions of same location)
- Group related backgrounds (indoor vs outdoor, by location)

### 4. Semantic Naming

**Best Practices:**
- Use descriptive, lowercase names with underscores
- Include location and type: `miguel_house_living_room`
- For outdoor scenes: `santa_cecilia_plaza_day`
- For generic scenes: `indoor_generic_hallway`

**Examples:**
```
scene_000 → miguel_house_bedroom
scene_001 → santa_cecilia_plaza_main
scene_002 → outdoor_alley_night
scene_003 → indoor_workshop
scene_004 → land_of_dead_marigold_bridge
```

### 5. Export Reviewed Results

Click **"💾 Export Reviewed Clusters"** button when done.

**Output Structure:**
```
scene_clusters_reviewed/
├── miguel_house_bedroom/
│   ├── frame_0001.png
│   ├── frame_0042.png
│   └── ...
├── santa_cecilia_plaza_main/
│   ├── frame_0023.png
│   └── ...
└── scene_review_report.json
```

**Review Report** contains:
- Timestamp of review
- Total clusters and images
- Original cluster mapping
- Modified cluster information

## Output Format

### Directory Structure

```
output_dir/
├── [cluster_name_1]/           # Renamed or original cluster name
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── [cluster_name_2]/
│   └── ...
└── scene_review_report.json    # Review metadata
```

### Review Report JSON

```json
{
  "timestamp": "2025-11-20T12:00:00",
  "original_cluster_dir": "/path/to/scene_clusters_giant",
  "total_clusters": 35,
  "total_images": 1850,
  "original_report": {
    "n_clusters": 38,
    "method": "kmeans",
    ...
  },
  "clusters": {
    "miguel_house_bedroom": {
      "display_name": "miguel_house_bedroom",
      "image_count": 48,
      "modified": true
    },
    ...
  }
}
```

## Integration with Pipeline

### Before Review

Scene clustering outputs from `scene_clustering.py`:
```
scene_clusters_giant/
├── scene_000/
├── scene_001/
├── ...
└── scene_clustering_report.json
```

### After Review

Reviewed and curated clusters ready for training:
```
scene_clusters_reviewed/
├── miguel_house_bedroom/      # Semantically named
├── santa_cecilia_plaza/       # Merged similar scenes
└── scene_review_report.json   # Tracking changes
```

### Next Steps

1. **Caption Generation**: Use reviewed clusters for VLM captioning
2. **Training Dataset Preparation**: Assemble final LoRA training data
3. **Configuration Generation**: Auto-generate training configs

## Tips and Best Practices

### Cluster Naming Conventions

**For Pixar Movies:**
- **Character-specific locations**: `{character}_house_{room}`
- **Public locations**: `{location_name}_{descriptor}`
- **Generic scenes**: `indoor_{type}` or `outdoor_{type}`

**Examples by Movie:**
- **Coco**: `miguel_house`, `santa_cecilia_plaza`, `land_of_dead`
- **Luca**: `luca_hideout`, `portorosso_piazza`, `underwater_kelp`
- **Turning Red**: `mei_house`, `toronto_streets`, `4town_concert`

### Merging Strategy

**Merge clusters when:**
- Same location, different camera angles
- Same scene, different times (day/night)
- Visually very similar backgrounds
- Total cluster size would be 30-100 images (optimal for training)

**Keep separate when:**
- Distinct locations or settings
- Different visual characteristics
- Different lighting conditions that matter for training
- Each cluster has 20+ images already

### Quality Control

**Before exporting, ensure:**
- All major scenes are well-represented (30+ images)
- Noise and low-quality clusters are deleted
- Similar scenes are merged to avoid redundancy
- All clusters have meaningful semantic names

## Troubleshooting

### Browser doesn't open automatically

```bash
# Manually open: http://localhost:5001
# Or use --no-browser flag and open URL yourself
python scripts/generic/clustering/interactive_scene_reviewer.py \
  --cluster-dir /path/to/clusters \
  --output-dir /path/to/output \
  --no-browser
```

### Port already in use

```bash
# Use a different port
bash scripts/batch/review_scene_clusters.sh turning-red 5002
```

### Images not loading

- Verify cluster directory exists and contains scene_* folders
- Check file permissions
- Ensure images are in supported formats (.png, .jpg, .jpeg)

## Advanced Usage

### Review Multiple Movies in Parallel

Open multiple reviewers on different ports:

```bash
# Terminal 1
bash scripts/batch/review_scene_clusters.sh turning-red 5001

# Terminal 2
bash scripts/batch/review_scene_clusters.sh luca 5002

# Terminal 3
bash scripts/batch/review_scene_clusters.sh up 5003
```

Access each at:
- http://localhost:5001 (Turning Red)
- http://localhost:5002 (Luca)
- http://localhost:5003 (Up)

### Batch Review Workflow

1. **Launch all reviewers** (different ports)
2. **Review in parallel** across different browsers/tabs
3. **Export each** when complete
4. **Verify outputs** using review reports
5. **Proceed to captioning** for all movies together

## Related Tools

- `scene_clustering.py` - Generates initial clusters
- `visualize_scene_clustering.py` - Creates cluster visualizations
- `prepare_training_data.py` - Assembles final training datasets
- `generate_training_configs.py` - Auto-generates LoRA configs

## Example Session

```bash
# 1. Check available movies
ls /mnt/data/ai_data/datasets/3d-anime/*/lora_data/scene_clusters_giant

# 2. Launch reviewer for Turning Red
bash scripts/batch/review_scene_clusters.sh turning-red 5001

# 3. In browser:
#    - Rename scene_000 → "mei_house_bedroom"
#    - Merge scene_003 into scene_001 (both are plaza scenes)
#    - Delete scene_037 (noise/low quality)
#    - Export reviewed clusters

# 4. Verify output
ls /mnt/data/ai_data/datasets/3d-anime/turning-red/lora_data/scene_clusters_reviewed
cat /mnt/data/ai_data/datasets/3d-anime/turning-red/lora_data/scene_clusters_reviewed/scene_review_report.json

# 5. Proceed to next movie
bash scripts/batch/review_scene_clusters.sh luca 5002
```

## Statistics and Reporting

The tool tracks:
- **Total Clusters**: Number of scene categories
- **Total Images**: Count of all background images
- **Average Cluster Size**: Images per cluster
- **Modified Clusters**: How many were edited/renamed

Review report includes original clustering metrics:
- Silhouette score
- Davies-Bouldin score
- Cluster size statistics (mean, std, min, max)

Use this data to assess clustering quality and make informed decisions about merging/splitting.

## Notes

- **Auto-save**: Changes are only saved when you click "Export"
- **Browser refresh**: Will reload original data, losing unsaved changes
- **Concurrent access**: Not recommended; one user per review session
- **Large datasets**: May be slow with 100+ clusters; consider reviewing in batches
