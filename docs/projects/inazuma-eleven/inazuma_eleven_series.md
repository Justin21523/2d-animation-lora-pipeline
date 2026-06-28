schema_version: "inazuma_series_v1.0"
doc_type: "series_profile"
franchise_id: "inazuma_eleven"
franchise_jp: "イナズマイレブン"
primary_media_scope:
  - "anime"
  - "games"
language: "en"
intended_use:
  - "program_parsing"
  - "animation_character_pipeline"
  - "lora_training"
  - "cross-character_comparison"
source_bundle:
  - "inazuma_eleven_series.md"
  - "character_Endou_Mamoru.md"
  - "character_Gouenji_Shuuya.md"
  - "character_Fudou_Akio.md"
  - "character_Utsunomiya_Toramaru.md"
  - "character_Matsukaze_Tenma.md"
  - "character_Inamori_Asuto.md"
  - "character_Nosaka_Yuuma.md"
build_notes:
  - "This file remodells the franchise around Worldview / Timeline / Mechanics / Character Archetypes; original wiki-style narrative chapter flow is intentionally not preserved."
---

# Inazuma Eleven — Franchise / World / Timeline Structured Profile

## 1. Overview (Series Positioning)

### 1.1 Core Purpose of the Series (What You Use This For)
- **Genre Core:** A youth competitive narrative framework combining *football × hissatsu techniques × exaggerated supernatural presentation*.
- **Narrative Engine:** A repeatable template of “team formation → confronting stronger systems/organizations → upgrading skills and beliefs → validation through major tournaments.”
- **Data Modeling Priority:** The same character appears repeatedly across different *timelines / parallel worlds / growth stages* (e.g., Original → GO +10 years, Ares/Orion alternate routes). Character versions must be split into distinct nodes to avoid cross-contamination during training.

### 1.2 Main Franchise Axes (For Worldview Tagging)
- **Original:** From school football to national and global stages (Football Frontier → external threats → FFI).
- **GO (Ten Years Later):** Football is systemically controlled by *Fifth Sector*; the central conflict is “reclaiming true football.”
- **Ares (Alternate Timeline):** The *Ares no Tenbin* plan and its data-driven football philosophy clash with the idea of “playing football for enjoyment.”
- **Orion (Parallel Extension):** The FFI is manipulated by the *Orion Foundation* using spies, referees, violence, and disguises.

---

## 2. World Structure (Machine-Parseable)

> This section models the world using parseable **Entities**, each with fixed fields:
> `entity_id / entity_type / timeline_scope / role / keywords`.

### 2.1 Core Entity Types
- `entity_type: team` (teams)
- `entity_type: tournament` (competitions / leagues / selections)
- `entity_type: organization` (organizations / systems / foundations / programs)
- `entity_type: mechanic` (supernatural systems: hissatsu, Keshin, Mixi Max, Soul)
- `entity_type: location` (schools, bases, training facilities)
- `entity_type: role_archetype` (character archetypes)

---

## 3. Timeline Model (Hard-Separated)

### 3.1 Timeline Variants
```yaml
timeline_variants:
  - timeline_id: original
    label: "Original Timeline"
    canonical_rule: "Main storyline (FF → Aliea → FFI)"
  - timeline_id: go
    label: "GO Timeline"
    canonical_rule: "New generation approx. 10 years after Season 3"
    time_offset_from_original: "+10y"
  - timeline_id: ares
    label: "Ares Timeline"
    canonical_rule: "Parallel timeline diverging from the original"
  - timeline_id: orion
    label: "Orion Timeline"
    canonical_rule: "Parallel extension of Ares (FFI under Orion control)"

```

### 3.2 Timeline Edges (Anti-Mixing Rules)

-   `original → go`: GO is explicitly defined as ~10 years after the end of Season 3.
    
-   `ares` is an _alternate timeline_: identical characters (e.g., Fudou Akio) exist as parallel versions.
    
-   `orion` extends `ares`: its main stage is the FFI under Orion Foundation manipulation.
    

----------

## 4. Canon Timeline Blocks (Program-Oriented Summaries)

> These are not plot summaries, but **conflict structures + systems/tournaments + character function slots**.

### 4.1 Original Timeline

```yaml
timeline_block:
  timeline_id: original
  primary_tournaments:
    - football_frontier
    - football_frontier_international
  conflict_layers:
    - "School team survival/rebuilding"
    - "Confrontation with expanding enemy forces"
    - "Escalation to international competition"
  key_entities:
    - team: raimon
    - organization: aliea_gakuen
    - team: inazuma_japan
  anchor_characters:
    - endou_mamoru

```

-   Season breakdown:  
    S1 = Football Frontier  
    S2 = Aliea Gakuen conflict  
    S3 = Football Frontier International (Inazuma Japan)
    

### 4.2 GO Timeline

```yaml
timeline_block:
  timeline_id: go
  time_offset_from_original: "+10y"
  primary_conflict: "Football controlled by Fifth Sector → reclaiming 'true football'"
  key_organizations:
    - fifth_sector
  anchor_characters:
    - matsukaze_tenma
    - endou_mamoru_go_coach
    - gouenji_shuuya_as_ishido

```

-   GO begins ~10 years after Season 3.
    
-   Protagonist: Matsukaze Tenma.
    
-   Raimon opposes Fifth Sector and its leader, Ishido Shuuji.
    
-   **Ishido Shuuji** is a masked identity of Gouenji Shuuya, serving as Holy Emperor and overseer.
    

### 4.3 Ares Timeline

```yaml
timeline_block:
  timeline_id: ares
  primary_conflict: "Ares system’s result-driven logic vs human-centered football"
  key_organizations:
    - ares_no_tenbin_program
  anchor_characters:
    - inamori_asuto
    - nosaka_yuuma

```

-   Narrative core: the Ares program is tied to child institutions and systematic talent creation.
    
-   Nosaka Yuuma initially relies on the Ares system for strategy and efficiency before acknowledging emotional value and enjoyment.
    

### 4.4 Orion Timeline

```yaml
timeline_block:
  timeline_id: orion
  primary_tournament:
    - football_frontier_international
  primary_conflict: "Orion Foundation manipulating match results"
  key_organizations:
    - orion_foundation
  anchor_characters:
    - inamori_asuto
    - nosaka_yuuma
    - endou_mamoru_orion_captain

```

-   Orion’s core mechanism: infiltration of teams with spies to control outcomes.
    

----------

## 5. Mechanics & Power Systems

### 5.1 Base Mechanic

```yaml
mechanic:
  mechanic_id: hissatsu
  label: "Hissatsu Techniques"
  scope: ["original","go","ares","orion"]
  categories:
    - shoot
    - dribble
    - block
    - catch
    - tactic

```

-   Hissatsu techniques form the core reusable asset across the franchise.
    

### 5.2 GO-Era Mechanics (Hard Visual Boundaries)

#### 5.2.1 Keshin (Avatars)

-   Explicitly listed in Tenma’s profile (e.g., Majin Pegasus).
    

#### 5.2.2 Keshin Fusion

-   Tenma’s Matei Gryphon is listed as a fusion form.
    

#### 5.2.3 Keshin Armed

-   Appears as an independent state in Chrono Stone.
    

#### 5.2.4 Mixi Max

-   Tenma’s Mixi Max partners (Shuu, King Arthur) define combination-based form assets.
    

#### 5.2.5 Soul

-   Soul exists as an independent visual/emotive layer (e.g., Toramaru’s Soul).
    

----------

## 6. Organization / System Graph

### 6.1 Fifth Sector

```yaml
entity:
  entity_id: fifth_sector
  entity_type: organization
  timeline_scope: ["go"]
  role: "Football control system; primary antagonist"
  keywords: ["control_soccer","holy_emperor","regulated_matches"]
  linked_character_versions:
    - "gouenji_shuuya@go::ishido_shuuji"

```

### 6.2 Ares no Tenbin Program

```yaml
entity:
  entity_id: ares_no_tenbin_program
  entity_type: organization
  timeline_scope: ["ares"]
  role: "Institutional talent cultivation system"
  keywords: ["training_program","institution_children","results_driven"]

```

### 6.3 Orion Foundation

```yaml
entity:
  entity_id: orion_foundation
  entity_type: organization
  timeline_scope: ["orion"]
  role: "Black-box FFI manipulation force"
  keywords: ["match_fixing","spies","disciples","referee_control"]

```

### 6.4 El Dorado / Protocol Omega

```yaml
entity:
  entity_id: el_dorado
  entity_type: organization
  timeline_scope: ["go"]
  role: "Time travel and historical interference"
  keywords: ["time_travel","timeline_interference"]
entity:
  entity_id: protocol_omega
  entity_type: organization
  timeline_scope: ["go"]
  role: "Violent football eradication unit"
  keywords: ["attack_style","timeline_attack","ragnarok_context"]

```

----------

## 7. Character Archetype Taxonomy

### 7.1 Archetype Library

```yaml
role_archetypes:
  - archetype_id: captain_heart
    function: "Emotional stabilizer and team core"
    common_tags: ["team_core","never_give_up","morale_boost"]
    examples:
      - "endou_mamoru@original"
  - archetype_id: ace_striker_icon
    function: "Visual and scoring symbol"
    common_tags: ["signature_shot","cool_facade","fire_element"]
    examples:
      - "gouenji_shuuya@original"
  - archetype_id: masked_antagonist_system
    function: "System representative with masked identity"
    common_tags: ["system_control","masked_identity","cold_authority"]
    examples:
      - "gouenji_shuuya@go::ishido_shuuji"
  - archetype_id: tactical_rebel
    function: "Strategic joker"
    common_tags: ["provocation","trick_play","tactical_joker"]
    examples:
      - "fudou_akio@original"
  - archetype_id: newgen_protagonist_breeze
    function: "Optimistic growth leader"
    common_tags: ["optimistic","growth_leader","wind_motif"]
    examples:
      - "matsukaze_tenma@go"
  - archetype_id: newgen_protagonist_sun
    function: "Parallel-line protagonist"
    common_tags: ["straightforward","fair_play","fire_element"]
    examples:
      - "inamori_asuto@ares_orion"
  - archetype_id: tactical_emperor
    function: "Cold strategic commander"
    common_tags: ["sharp_tactics","calm","ruthless_when_needed"]
    examples:
      - "nosaka_yuuma@ares_orion"

```

----------

## 8. Visual Style Change Model

### 8.1 Cross-Series Visual Axes

-   `uniform_axis`: school / national / coach / suit identities.
    
-   `powerform_axis`: Keshin / Armed / Mixi Max / Soul as separate datasets.
    
-   `timeline_axis`: same-name characters across timelines **must not** share assets.
    

### 8.2 Hard No-Mix Rules

```yaml
no_mix_rules:
  - rule_id: no_mix_timeline_versions
    description: "Same character across timelines must be separated"
  - rule_id: no_mix_go_powerforms
    description: "Keshin / Armed / Mixi Max / Soul are separate datasets"
  - rule_id: no_mix_mask_identity
    description: "Masked identities and true identities must not be merged"

```

----------

## 9. Animation / AI Training Recommendations

### 9.1 Dataset Partition Plan

```yaml
dataset_partitions:
  - partition_id: original_base
    include: ["timeline_original","no_keshin","no_soul"]
  - partition_id: go_regulation_arc
    include: ["timeline_go","fifth_sector","holy_road_context"]
  - partition_id: go_powerform_layer
    include: ["timeline_go","keshin|keshin_armed|miximax|soul"]
  - partition_id: ares_system_arc
    include: ["timeline_ares","ares_no_tenbin_program"]
  - partition_id: orion_corruption_arc
    include: ["timeline_orion","orion_foundation","ffi_context"]

```

### 9.2 Caption Schema

```yaml
caption_schema:
  required_fields:
    - franchise_id
    - timeline_id
    - entity_character_id
    - form_state
    - uniform_state
    - position_role
    - element_tag
  optional_fields:
    - hissatsu_tags
    - organization_context
    - emotion_tags
    - interaction_tags

```

----------

## 10. Minimal Known-True Facts Index

-   Original: S1 = FF, S2 = Aliea, S3 = FFI + Inazuma Japan.
    
-   GO: ~10 years later; protagonist Matsukaze Tenma; Fifth Sector conflict.
    
-   Ares: Explicit alternate timeline.
    
-   Orion: Orion Foundation manipulates FFI.
    
-   GO mechanics: Keshin / Fusion / Armed / Mixi Max / Soul are independent form layers.
    

