# Pixar-Style 3D Animation Prompt Templates Reference

This document contains **276 detailed prompt templates** for generating high-quality Pixar-style 3D character images. These templates are designed for use with:
- Stable Diffusion XL (SDXL)
- Character Identity LoRAs
- Action/Pose/Expression LoRA training

## Template Statistics

| Category | Count | Token Range |
|----------|-------|-------------|
| **Pose** | 66 | 60-80 tokens |
| **Expression** | 67 | 60-80 tokens |
| **Action** | 143 | 60-80 tokens |
| **Total** | **276** | - |

## How to Use

### Template Format
Each template uses two placeholders:
- `{character}` - Character name and description (e.g., "alberto (Alberto from Pixar Luca, Italian teen, curly brown hair)")
- `{style}` - Style prefix (e.g., "3d animation, pixar style, high quality, detailed")

### Example Usage in WebUI
```
alberto (Alberto from Pixar Luca, Italian teen, curly brown hair), 3d animation, pixar style, high quality, detailed, [TEMPLATE CONTENT HERE], pixar style 3d animated character
```

### Recommended Negative Prompt
```
multiple people, two people, group, crowd, duplicate character, duplicate person, clone, second person, extra character, photograph, photo, realistic, photorealistic, real person, real life, live action, hyperrealistic, real human, candid photo, portrait photo, stock photo, adult, elderly, old person, mature, grown-up, old man, baby, toddler, infant, young child, girl, female, woman, extra limbs, extra arms, extra legs, extra hands, extra fingers, missing limbs, missing arms, missing legs, missing hands, missing fingers, deformed, disfigured, distorted, malformed, mutated, mutation, bad anatomy, wrong anatomy, anatomically incorrect, blurry, out of focus, unfocused, fuzzy, hazy, soft focus, low quality, bad quality, worst quality, low resolution, low res, jpeg artifacts, compression artifacts, pixelated, grainy, noisy, watermark, text, signature, username, artist name, bad proportions, gross proportions, unnatural proportions
```

---

## POSE TEMPLATES (66)

### Standing Poses (15)

1. **Welcoming Open Pose**
```
welcoming open pose with arms spread wide in friendly greeting gesture, weight balanced with slight forward lean showing approachability, warm genuine smile with crinkled eyes, bright cheerful lighting with soft wraparound quality, friendly social environment, inviting atmosphere that draws viewer in, casual comfortable clothing with relaxed fabric draping
```

2. **Neutral Standing**
```
full body view standing straight in natural relaxed posture with weight evenly distributed on both feet, arms hanging loosely at sides with slight natural curve, neutral calm expression with soft gaze directed at camera, professional studio environment with seamless backdrop, three-point lighting setup with soft key light creating gentle shadows and subtle rim light separating figure from background, smooth skin rendering with subsurface scattering, detailed fabric textures on clothing
```

3. **Protective Guarded Stance**
```
protective guarded stance with arms wrapped around self in self-comforting embrace, shoulders slightly hunched and body angled away, vulnerable uncertain expression with downcast eyes, soft diffused lighting with cool blue undertones creating somber mood, quiet isolated environment, emotionally sensitive moment captured with care, subtle body language details showing psychological state
```

4. **Confident Power Pose**
```
confident power stance with hands firmly planted on hips, chest open and shoulders back showing self-assurance, direct assertive eye contact with slight confident smile, strong dramatic lighting with bold shadows emphasizing form, professional or leadership environment, commanding presence atmosphere, business casual or formal attire with clean lines
```

5. **Casual Lean**
```
casual leaning pose with shoulder against wall or surface, weight shifted to one side with relaxed hip angle, easygoing friendly expression with slight smirk, natural ambient lighting suggesting everyday setting, urban or indoor casual environment, laid-back comfortable atmosphere, streetwear or casual clothing with natural draping
```

6. **Attentive Ready Stance**
```
attentive ready stance with feet shoulder-width apart, arms loose at sides but engaged, alert focused expression ready for action, bright even lighting suggesting active environment, sports or activity setting, energetic anticipatory atmosphere, athletic wear or activity-appropriate clothing
```

7. **Contemplative Stance**
```
contemplative standing pose with weight shifted to one leg, one hand raised thoughtfully to chin, pensive introspective expression gazing into distance, soft moody lighting with gentle gradients, quiet reflective environment, thoughtful meditative atmosphere, comfortable casual clothing
```

8. **Dynamic Contrapposto**
```
classical contrapposto stance with weight on one leg creating natural S-curve in body, opposite shoulder and hip tilted, relaxed elegant expression, artistic studio lighting with renaissance-inspired quality, timeless aesthetic environment, graceful balanced atmosphere, flowing clothing that follows body lines
```

9. **Arms Crossed Confident**
```
confident stance with arms crossed over chest, feet planted firmly shoulder-width apart, assured knowing expression with raised eyebrow, professional lighting setup with clean shadows, business or formal environment, self-assured authoritative atmosphere, professional attire with crisp details
```

10. **Parade Rest Formal**
```
formal parade rest position with feet apart and hands clasped behind back, spine straight with chin level, dignified composed expression, even institutional lighting, formal ceremonial environment, respectful disciplined atmosphere, uniform or formal clothing
```

11. **Relaxed Asymmetric**
```
relaxed asymmetric stance with weight shifted left, one knee slightly bent, casual at-ease expression, natural soft ambient lighting, everyday casual environment, comfortable relaxed atmosphere, casual everyday clothing
```

12. **Hip Pop Casual**
```
casual hip pop stance with weight dramatically shifted creating curved body line, hand possibly resting on raised hip, playful confident expression, bright fashion-forward lighting, trendy urban environment, stylish confident atmosphere, fashionable clothing with personality
```

13. **Waiting Patient**
```
patient waiting pose with hands loosely clasped in front, weight evenly distributed, calm expectant expression, soft diffused lighting suggesting indoor waiting area, public space environment, quiet patient atmosphere, appropriate casual or formal attire
```

14. **Heroic Wide Stance**
```
heroic wide power stance with feet spread beyond shoulder width, fists at sides or on hips, determined fierce expression, dramatic uplighting creating heroic silhouette, epic adventure environment, powerful inspiring atmosphere, hero or adventure attire
```

15. **Shy Withdrawn**
```
shy withdrawn stance with shoulders slightly forward, arms close to body, timid uncertain expression with eyes looking down or away, soft gentle lighting with muted tones, quiet intimate environment, vulnerable delicate atmosphere, modest unassuming clothing
```

### Sitting Poses (15)

16. **Relaxed Seated**
```
relaxed seated pose in comfortable chair with back against support, one leg possibly crossed over other, content peaceful expression, warm cozy lighting suggesting home environment, living room or comfortable indoor setting, restful comfortable atmosphere, casual loungewear or comfortable clothing
```

17. **Engaged Forward Lean**
```
engaged seated pose leaning forward with elbows on knees, attentive interested expression with focused eye contact, practical even lighting suggesting meeting or conversation, office or social setting, active engaged atmosphere, smart casual attire
```

18. **Cross-Legged Floor**
```
cross-legged seated pose on floor or cushion, hands resting on knees or in lap, serene meditative expression, soft peaceful lighting with warm tones, zen or peaceful indoor environment, calm centered atmosphere, comfortable loose clothing
```

19. **Elegant Side Saddle**
```
graceful side-saddle seated pose with legs positioned elegantly to one side, refined upright posture with hands folded in lap, poised sophisticated expression, soft glamorous lighting with gentle highlights and shadows, elegant refined environment, classic timeless aesthetic, formal feminine attire with flowing fabric details
```

20. **Perched Alert**
```
attentive seated pose perched on edge of seat leaning forward with engaged interest, hands on knees in ready position, alert curious expression with wide attentive eyes, bright even lighting suggesting active social or educational setting, engaged learning or meeting environment, participatory atmosphere showing active listening, neat casual clothing appropriate for social setting
```

21. **Casual Sprawl**
```
relaxed lounging pose leaning back in chair with casual sprawl, one leg extended and other bent, completely at-ease expression with lazy contentment, warm ambient lighting suggesting comfortable evening setting, cozy living space environment, restful leisure atmosphere, comfortable loungewear with relaxed draping
```

22. **Thoughtful Hunched**
```
contemplative hunched seated pose with elbows on knees and chin resting on hands, deeply thoughtful expression with distant gaze, moody atmospheric lighting with shadows, quiet private space environment, introspective reflective atmosphere, casual comfortable clothing
```

23. **Proper Upright**
```
proper formal seated pose with straight spine and feet flat on floor, hands folded neatly in lap, polite attentive expression, clean professional lighting, formal office or interview environment, respectful professional atmosphere, business formal attire
```

24. **Desk Working**
```
focused working pose seated at desk with hands on keyboard or papers, body leaning slightly toward work, concentrated productive expression, practical task lighting from desk lamp, home office or workspace environment, busy productive atmosphere, smart casual work attire
```

25. **Knee Hug Cozy**
```
cozy protective pose with knees pulled up to chest while seated, arms wrapped around legs, content comfortable expression, warm soft lighting suggesting evening, cozy intimate setting like bed or couch, safe comfortable atmosphere, soft comfortable sleepwear or loungewear
```

### Ground Poses (10)

26. **Supine Relaxed**
```
relaxed supine pose lying on back with arms at sides or behind head, legs extended comfortably, peaceful resting expression with closed or half-closed eyes, soft overhead lighting simulating sky or ceiling, outdoor grass or indoor floor environment, completely relaxed restful atmosphere, casual comfortable clothing
```

27. **Side Lying Casual**
```
casual side-lying pose with head propped on hand, elbow on ground, other arm draped over body, relaxed friendly expression, warm intimate lighting, bedroom or living room floor environment, comfortable casual atmosphere, loungewear or casual clothing
```

28. **Prone Playful**
```
playful prone pose lying on stomach with chin resting on hands, feet possibly kicked up behind, cheerful engaging expression looking at camera, bright natural lighting, outdoor grass or indoor carpet environment, lighthearted fun atmosphere, casual playful clothing
```

29. **Fetal Protective**
```
protective fetal position curled on side with knees drawn to chest, arms wrapped around self, peaceful or vulnerable expression, soft gentle lighting with cool tones, bed or soft surface environment, safe protected atmosphere, soft comfortable clothing
```

30. **Starfish Open**
```
open relaxed starfish pose lying flat with arms and legs spread wide, completely relaxed expression with eyes closed, even overhead lighting, bed or soft floor environment, total relaxation atmosphere, sleepwear or minimal comfortable clothing
```

### Dynamic Movement Poses (12)

31. **Walking Natural**
```
natural walking pose mid-stride with one foot forward, arms swinging in opposition, relaxed pleasant expression, outdoor natural lighting with sun direction, sidewalk or path environment, everyday casual motion, comfortable walking attire
```

32. **Running Athletic**
```
dynamic running pose at full stride with arms pumping and legs extended, focused determined expression, bright action lighting capturing motion, outdoor track or path environment, energetic athletic atmosphere, proper running attire and footwear
```

33. **Jumping Joy**
```
joyful jumping pose at peak height with arms raised overhead, legs bent or split, ecstatic happy expression with wide smile, bright uplifting lighting from below, open outdoor or studio environment, celebratory joyful atmosphere, casual clothing showing movement
```

34. **Dancing Flow**
```
flowing dance pose with body in graceful curved position, arms extended expressively, passionate artistic expression, dramatic stage lighting with color, dance studio or stage environment, creative expressive atmosphere, dance attire allowing movement
```

35. **Stretching Reach**
```
stretching pose reaching upward with fully extended arms and body, standing on toes with elongated form, refreshed energized expression, bright morning lighting, bedroom or exercise space environment, awakening energized atmosphere, sleepwear or athletic wear
```

36. **Crouching Ready**
```
athletic crouching pose with bent knees and lowered center of gravity, arms ready for action, alert focused expression, dynamic action lighting, sports field or action environment, tense ready-to-spring atmosphere, athletic or action-appropriate clothing
```

37. **Twisting Turn**
```
dynamic twisting pose with torso rotated while hips face forward, arms following rotation, engaged active expression, motion-capturing lighting, open movement space environment, dynamic energetic atmosphere, flexible clothing allowing twist
```

38. **Balancing One Leg**
```
careful balancing pose on one leg with other raised, arms extended for stability, concentrated focused expression, stable even lighting, yoga studio or calm environment, centered balanced atmosphere, fitted exercise clothing
```

39. **Leaping Forward**
```
forward leaping pose with body fully extended horizontally, one leg leading, arms reaching forward, determined confident expression, dynamic action lighting capturing trajectory, outdoor or sports environment, athletic atmosphere, sport-appropriate attire
```

40. **Spinning Motion**
```
mid-spin pose with body rotating, clothing and hair showing movement, joyful carefree expression, circular motion lighting effect, dance floor or open space environment, playful dynamic atmosphere, flowing clothing enhancing spin visual
```

### Camera Angle Variations (14)

41. **Front Full Body**
```
full body front view standing naturally with arms relaxed at sides, direct eye contact with camera, symmetrical centered composition, even studio lighting from front, clean backdrop environment, professional portrait atmosphere, appropriate complete outfit visible
```

42. **Three-Quarter Dynamic**
```
three-quarter view showing depth and dimension with body angled, head turned toward camera, engaging dynamic expression, sculpted lighting emphasizing form, studio or contextual environment, dimensional interesting atmosphere, clothing showing form and detail
```

43. **Profile Classic**
```
classic profile side view with clean silhouette, chin level with dignified posture, serene or thoughtful expression, rim lighting creating separation from background, simple elegant environment, timeless portrait atmosphere, clothing silhouette clearly defined
```

44. **Back Over-Shoulder**
```
back view with head turned over shoulder toward camera, mysterious or inviting expression, dramatic rim lighting from front, atmospheric environment, intriguing narrative atmosphere, back details of clothing visible
```

45. **Low Angle Heroic**
```
dramatic low angle full body shot looking up at standing figure creating powerful imposing presence, figure looming large in frame, confident commanding expression looking down at camera, strong heroic lighting from above and behind, epic cinematic atmosphere, powerful dominant composition, detailed costume or outfit rendered from below perspective
```

46. **High Angle Intimate**
```
gentle high angle shot looking down at figure creating approachable intimate perspective, figure appearing smaller and more vulnerable, soft open expression looking up at camera, diffused overhead lighting creating soft shadows, personal intimate atmosphere, unique overhead composition showing top details of outfit and hair
```

47. **Dutch Angle Dynamic**
```
tilted dutch angle view creating dynamic tension and visual interest, body positioned diagonally in frame, engaging energetic expression, dramatic off-kilter lighting, stylized artistic environment, creative unconventional atmosphere, clothing and pose enhanced by angle
```

48. **Extreme Close-Up Face**
```
extreme close-up focusing on face and upper shoulders, highly detailed facial features and skin texture, intense emotional expression with visible micro-expressions, soft beauty lighting with catchlights in eyes, intimate personal atmosphere, minimal clothing visible but high detail on what shows
```

49. **Wide Environmental**
```
wide shot placing figure small within larger environment, body language readable from distance, contextual expression fitting scene, environmental lighting natural to setting, rich detailed background environment, storytelling narrative atmosphere, full outfit visible in context
```

50. **Medium Shot Classic**
```
classic medium shot from waist up showing upper body and hands, natural conversational framing, friendly engaging expression, balanced flattering lighting, neutral or contextual background, approachable relatable atmosphere, upper body clothing and accessories detailed
```

### Emotional/Psychological Poses (16)

51. **Victorious Triumph**
```
triumphant victory pose with arms raised high in celebration, body open and expanded with pride, elated joyful expression of accomplishment, bright celebratory lighting with golden tones, achievement or competition environment, peak emotional moment atmosphere, appropriate attire for victory context
```

52. **Defeated Dejected**
```
dejected defeated pose with slumped shoulders and lowered head, arms hanging limply or wrapped around self, sorrowful disappointed expression, dim muted lighting with cool tones, quiet aftermath environment, emotional low point atmosphere, rumpled or disheveled clothing
```

53. **Anxious Nervous**
```
nervous anxious pose with tense shoulders and fidgeting hands, weight shifting uncertainly, worried apprehensive expression with furrowed brow, slightly harsh unflattering lighting, stressful uncertain environment, tense uncomfortable atmosphere, clothing possibly being nervously adjusted
```

54. **Peaceful Serene**
```
serene peaceful pose with relaxed open body language, hands possibly in gentle mudra or resting softly, tranquil blissful expression with soft smile, soft ethereal lighting with warm glow, calm natural or spiritual environment, deeply peaceful atmosphere, flowing comfortable peaceful clothing
```

55. **Angry Aggressive**
```
aggressive angry pose with tensed muscles and clenched fists, body leaning forward confrontationally, furious intense expression with bared teeth, harsh dramatic lighting with strong shadows, confrontational charged environment, volatile intense atmosphere, clothing stretched tight with tension
```

56. **Curious Investigating**
```
curious investigating pose leaning forward with head tilted, hand possibly raised to chin or touching object of interest, intrigued fascinated expression with wide attentive eyes, focused spotlight lighting on subject of curiosity, discovery exploration environment, engaged inquisitive atmosphere, practical appropriate clothing
```

57. **Proud Accomplished**
```
proud accomplished pose with lifted chin and squared shoulders, hands on hips or holding achievement, satisfied proud expression with confident smile, warm accomplishment lighting, recognition or display environment, celebratory proud atmosphere, appropriate formal or achievement-related attire
```

58. **Lonely Isolated**
```
lonely isolated pose with body turned partially away, arms possibly self-embracing, melancholic wistful expression, single source lighting creating long shadows, empty sparse environment, solitary reflective atmosphere, simple unadorned clothing
```

59. **Playful Mischievous**
```
playful mischievous pose with body in active ready position, hands possibly behind back hiding something, impish knowing expression with sly smile, bright playful lighting, fun casual environment, lighthearted sneaky atmosphere, casual playful clothing
```

60. **Determined Focused**
```
determined focused pose with squared stance and purposeful positioning, hands possibly clenched or ready, intense concentrated expression, strong directional lighting emphasizing determination, goal-oriented achievement environment, driven motivated atmosphere, practical action-ready clothing
```

61. **Surprised Startled**
```
surprised startled pose with body slightly jerked back, hands raised in shock, wide-eyed amazed expression with open mouth, sudden bright lighting suggesting flash or revelation, unexpected encounter environment, moment of shock atmosphere, clothing slightly disheveled from sudden movement
```

62. **Loving Tender**
```
tender loving pose with open welcoming arms or gentle embrace gesture, body soft and approachable, warm loving expression with gentle smile, soft romantic lighting with warm tones, intimate personal environment, affectionate caring atmosphere, soft comfortable clothing
```

63. **Exhausted Tired**
```
exhausted tired pose with drooping posture and heavy limbs, possibly leaning on support, weary fatigued expression with half-closed eyes, dim low-energy lighting, end of long day environment, depleted drained atmosphere, loosened or disheveled clothing
```

64. **Hopeful Optimistic**
```
hopeful optimistic pose with body oriented upward or forward, hands possibly clasped in hope, bright expectant expression looking toward future, inspiring uplifting lighting with golden tones, new beginning or horizon environment, positive anticipatory atmosphere, fresh clean clothing
```

65. **Embarrassed Shy**
```
embarrassed shy pose with turned away body and covered face, shoulders hunched protectively, blushing flustered expression, soft diffused lighting hiding harsh details, awkward social environment, uncomfortable self-conscious atmosphere, modest covering clothing
```

66. **Confident Swagger**
```
confident swagger pose with loose relaxed walk stance, weight shifted with casual lean, self-assured smirking expression, cool dramatic lighting, urban or social environment, effortlessly cool atmosphere, stylish trendy clothing
```

---

## EXPRESSION TEMPLATES (67)

### Happy/Joy Expressions (10)

1. **Genuine Warm Smile**
```
close-up portrait with genuine warm smile reaching the eyes, natural crow's feet from real happiness, relaxed facial muscles with soft cheeks, bright warm lighting enhancing positive mood, cheerful atmosphere, detailed skin rendering with natural subsurface scattering showing healthy glow
```

2. **Laughing Heartily**
```
upper body shot capturing moment of hearty laughter, mouth open showing teeth, eyes squeezed with joy, head tilted back slightly, bright natural lighting, joyful exuberant atmosphere, detailed rendering of laugh lines and dimples
```

3. **Beaming Radiant**
```
beaming radiant smile with maximum happiness, eyes sparkling with delight, full teeth showing smile, slightly raised cheeks creating apple shapes, golden warm lighting, euphoric joyful atmosphere, flawless skin rendering with healthy luminosity
```

4. **Subtle Content Smile**
```
subtle content smile with lips gently curved upward, peaceful serene eyes with soft gaze, relaxed facial expression showing inner peace, soft diffused lighting, calm satisfied atmosphere, smooth skin with natural subtle highlights
```

5. **Excited Enthusiastic**
```
excited enthusiastic expression with wide bright eyes, open smile with raised eyebrows, animated energetic facial muscles, dynamic bright lighting, thrilling exciting atmosphere, detailed expression showing genuine excitement
```

6. **Amused Chuckling**
```
amused chuckling expression with slight smile and crinkled eyes, one eyebrow possibly raised, knowing amused look, warm friendly lighting, lighthearted fun atmosphere, natural expression lines showing humor
```

7. **Delighted Surprised Joy**
```
delighted expression showing pleasant surprise, wide eyes with raised eyebrows, open smile emerging, hands possibly raised to face, bright revealing lighting, wonderful surprise atmosphere, detailed shocked-to-happy transition
```

8. **Peaceful Blissful**
```
peaceful blissful expression with eyes closed and serene smile, completely relaxed facial features, ethereal transcendent quality, soft glowing lighting, meditative peaceful atmosphere, smooth idealized skin rendering
```

9. **Proud Satisfied**
```
proud satisfied expression with confident slight smile, chin slightly raised, eyes showing accomplishment, dignified composed features, warm achievement lighting, successful triumphant atmosphere, detailed self-assured expression
```

10. **Playful Teasing**
```
playful teasing expression with mischievous smile, one eyebrow raised knowingly, twinkling eyes suggesting fun, bright playful lighting, lighthearted joking atmosphere, detailed impish expression quality
```

### Sad/Melancholy Expressions (8)

11. **Deeply Sorrowful**
```
deeply sorrowful expression with downturned mouth and sad eyes, furrowed brow showing emotional pain, glistening eyes suggesting held-back tears, soft muted lighting with blue tones, melancholic heavy atmosphere, detailed rendering of emotional distress
```

12. **Quietly Sad**
```
quietly sad expression with subtle downturn of features, distant unfocused gaze, slight frown barely visible, subdued gentle lighting, contemplative sad atmosphere, natural understated emotional rendering
```

13. **Crying Tearful**
```
tearful crying expression with tears streaming down cheeks, red-rimmed eyes and wet lashes, trembling lip and scrunched features, soft sympathetic lighting, deeply emotional atmosphere, highly detailed tear and moisture rendering
```

14. **Disappointed Let Down**
```
disappointed let-down expression with deflated features, lowered eyebrows and turned-down mouth, eyes showing hurt and disappointment, flat unflattering lighting, crushing disappointment atmosphere, detailed defeated expression
```

15. **Wistful Longing**
```
wistful longing expression gazing into distance, soft sad eyes with distant focus, slightly parted lips in gentle sigh, golden nostalgic lighting, yearning romantic atmosphere, dreamy soft-focus quality with detailed eyes
```

16. **Grieving Bereft**
```
grieving bereft expression with deeply pained features, closed or nearly closed eyes, mouth tight with suppressed emotion, dim heavy lighting, profound loss atmosphere, detailed rendering of deep sorrow
```

17. **Melancholic Pensive**
```
melancholic pensive expression with thoughtful sad eyes, slight frown of contemplation, head possibly tilted in reflection, moody atmospheric lighting, introspective sad atmosphere, artistic emotional rendering
```

18. **Bittersweet Memory**
```
bittersweet expression with sad eyes but small smile, conflicted emotional display, nostalgic distant gaze, warm but fading lighting, mixed emotion atmosphere, complex nuanced expression rendering
```

### Angry/Frustrated Expressions (8)

19. **Furiously Angry**
```
furiously angry expression with bared teeth and flared nostrils, deeply furrowed brow and narrowed eyes, tensed jaw and neck muscles, harsh dramatic lighting with strong shadows, explosive rage atmosphere, intense detailed anger rendering
```

20. **Quietly Seething**
```
quietly seething expression with tightened jaw and cold eyes, controlled but visible anger in features, slight nostril flare, cool harsh lighting, suppressed fury atmosphere, subtle but intense anger details
```

21. **Frustrated Exasperated**
```
frustrated exasperated expression with tired angry eyes, furrowed brow showing stress, mouth possibly open in exclamation, uneven stressful lighting, overwhelming frustration atmosphere, detailed stress expression
```

22. **Annoyed Irritated**
```
annoyed irritated expression with slight eye roll or narrowed gaze, tight-lipped displeasure, one eyebrow possibly raised, flat unflattering lighting, minor annoyance atmosphere, subtle irritation details
```

23. **Indignant Offended**
```
indignant offended expression with raised chin and narrowed eyes, tight pressed lips showing displeasure, shocked and angry combination, bright revealing lighting, righteous anger atmosphere, detailed offense expression
```

24. **Scowling Dark**
```
dark scowling expression with heavily furrowed brow, deeply turned-down mouth, shadowed intimidating eyes, dramatic low-key lighting, threatening menacing atmosphere, intense shadow rendering on features
```

25. **Contemptuous Disdain**
```
contemptuous disdainful expression with one lip raised in sneer, narrowed dismissive eyes, slightly raised chin, cold harsh lighting, superior disgusted atmosphere, detailed sneer rendering
```

26. **Bitter Resentful**
```
bitter resentful expression with hardened eyes and set jaw, deep frown lines visible, overall hardened appearance, unflattering stark lighting, long-held anger atmosphere, weathered detailed expression
```

### Surprised/Shocked Expressions (6)

27. **Completely Shocked**
```
completely shocked expression with wide eyes and dropped jaw, raised eyebrows at maximum height, frozen stunned features, bright sudden lighting, total surprise atmosphere, extreme expression detail
```

28. **Pleasantly Surprised**
```
pleasantly surprised expression with wide eyes and emerging smile, raised eyebrows with delight, open happy mouth, warm bright lighting, wonderful surprise atmosphere, detailed joy-shock combination
```

29. **Startled Alarmed**
```
startled alarmed expression with fear-widened eyes, sharply raised eyebrows, tensed facial muscles, sudden harsh lighting, immediate danger atmosphere, detailed fight-or-flight response
```

30. **Mildly Surprised**
```
mildly surprised expression with slightly raised eyebrows, small o-shaped mouth, widened but not extreme eyes, even natural lighting, casual surprise atmosphere, subtle surprised details
```

31. **Disbelieving Incredulous**
```
disbelieving incredulous expression with skeptically raised eyebrow, open mouth in disbelief, head possibly tilted or pulled back, revealing bright lighting, impossible situation atmosphere, detailed doubt expression
```

32. **Awestruck Wonder**
```
awestruck expression of pure wonder with wide starry eyes, slightly open mouth in amazement, raised eyebrows with reverence, dramatic upward lighting, magnificent discovery atmosphere, detailed wonder rendering
```

### Fear/Worry Expressions (6)

33. **Terrified Fearful**
```
terrified fearful expression with extremely wide eyes showing whites, raised tight eyebrows, open mouth possibly screaming, harsh dramatic lighting with shadows, imminent threat atmosphere, intense fear detail rendering
```

34. **Anxious Worried**
```
anxious worried expression with tight features and darting eyes, furrowed worried brow, lips pressed together nervously, unstable flickering lighting, uncertain stressful atmosphere, detailed anxiety expression
```

35. **Nervous Apprehensive**
```
nervous apprehensive expression with wide uncertain eyes, slightly furrowed brow, tight attempted smile, slightly harsh lighting, anticipating bad news atmosphere, subtle nervousness details
```

36. **Paranoid Suspicious**
```
paranoid suspicious expression with narrowed darting eyes, tense facial muscles, tight-lipped wariness, shadowy uncertain lighting, threatening paranoid atmosphere, detailed suspicious rendering
```

37. **Vulnerable Scared**
```
vulnerable scared expression with wide pleading eyes, raised worried eyebrows, trembling lip, soft but ominous lighting, helpless endangered atmosphere, detailed vulnerability rendering
```

38. **Panicked Desperate**
```
panicked desperate expression with wild wide eyes, open gasping mouth, raised distressed eyebrows, chaotic harsh lighting, life-or-death urgency atmosphere, extreme panic detail
```

### Thoughtful/Contemplative Expressions (6)

39. **Deep in Thought**
```
deeply thoughtful expression with focused distant gaze, slightly furrowed contemplative brow, relaxed but engaged features, soft thinking lighting, intellectual concentration atmosphere, detailed thought expression
```

40. **Curious Wondering**
```
curious wondering expression with engaged bright eyes, slightly raised interested eyebrows, small interested smile, warm inquisitive lighting, discovery exploration atmosphere, detailed curiosity rendering
```

41. **Skeptical Doubting**
```
skeptical doubting expression with one raised eyebrow, slightly narrowed assessing eyes, corner of mouth turned down, cool analytical lighting, questioning doubt atmosphere, detailed skepticism rendering
```

42. **Confused Puzzled**
```
confused puzzled expression with furrowed brow and squinted eyes, head tilted in confusion, slight frown of bewilderment, even but unhelpful lighting, perplexing mystery atmosphere, detailed confusion rendering
```

43. **Concentrating Focused**
```
intensely concentrating expression with narrowed focused eyes, tightened jaw showing effort, furrowed brow of mental exertion, targeted task lighting, demanding concentration atmosphere, detailed focus rendering
```

44. **Remembering Nostalgic**
```
remembering nostalgic expression with soft distant gaze, gentle half-smile, slightly raised eyebrows in recollection, warm golden lighting, past memory atmosphere, detailed nostalgia rendering
```

### Neutral/Calm Expressions (5)

45. **Completely Neutral**
```
completely neutral expression with relaxed balanced features, direct steady gaze, no particular emotion showing, even professional lighting, calm baseline atmosphere, perfectly balanced rendering
```

46. **Serene Peaceful**
```
serene peaceful expression with soft calm eyes, gentle slight smile, completely relaxed features, soft ethereal lighting, transcendent calm atmosphere, smooth idealized rendering
```

47. **Patient Waiting**
```
patient waiting expression with calm attentive eyes, relaxed neutral mouth, composed steady features, soft ambient lighting, patient expectant atmosphere, subtle attention rendering
```

48. **Alert Attentive**
```
alert attentive expression with bright focused eyes, slightly raised interested eyebrows, ready engaged features, clear bright lighting, active attention atmosphere, detailed alertness rendering
```

49. **Composed Dignified**
```
composed dignified expression with steady confident gaze, slight formal smile, refined controlled features, elegant portrait lighting, sophisticated formal atmosphere, refined expression rendering
```

### Complex/Mixed Expressions (18)

50. **Bittersweet Smile**
```
bittersweet smile expression with upturned lips but sad eyes, complex mixed emotion visible, hint of both joy and sorrow, soft golden lighting, emotionally complex atmosphere, nuanced detailed rendering
```

51. **Nervous Excitement**
```
nervous excitement expression with wide eager eyes showing slight fear, tight smile of anticipation, raised eyebrows of uncertainty, dynamic bright lighting, thrilling anxiety atmosphere, complex emotion rendering
```

52. **Proud but Humble**
```
proud but humble expression with satisfied eyes but modest smile, slight bow of head with achievement visible, balanced dignity and humility, warm accomplishment lighting, gracious success atmosphere, nuanced pride rendering
```

53. **Lovingly Concerned**
```
lovingly concerned expression with worried but caring eyes, gentle frown of worry, soft caring features, warm protective lighting, caring worry atmosphere, detailed compassion rendering
```

54. **Amused Disbelief**
```
amused disbelief expression with laughing eyes but skeptical eyebrow, smile fighting with doubt, head shake visible in features, bright revealing lighting, can't-believe-it atmosphere, complex humor rendering
```

55. **Determined Despite Fear**
```
brave determined expression showing underlying fear, set jaw with slightly wide eyes, courage overcoming terror, dramatic heroic lighting, courageous moment atmosphere, complex bravery rendering
```

56. **Forced Fake Smile**
```
forced fake smile expression with upturned mouth but unhappy eyes, visible strain in expression, disconnect between features, slightly harsh lighting, social pretense atmosphere, detailed fake emotion rendering
```

57. **Tearful Joy**
```
tearful joy expression with tears streaming but huge smile, overwhelmed happy eyes, quivering happy features, bright warm lighting, overwhelming happiness atmosphere, detailed joyful tears rendering
```

58. **Reluctant Agreement**
```
reluctant agreement expression with slight eye roll and small smile, giving-in visible in features, mix of annoyance and acceptance, even neutral lighting, compromise reached atmosphere, complex acceptance rendering
```

59. **Curious but Cautious**
```
curious but cautious expression with interested but wary eyes, leaning forward but ready to retreat, fascination mixed with wariness, careful revealing lighting, uncertain discovery atmosphere, complex interest rendering
```

60. **Impressed Grudgingly**
```
grudgingly impressed expression with raised eyebrows despite tight lips, fighting smile visible, reluctant admiration showing, cool but brightening lighting, unexpected respect atmosphere, complex admiration rendering
```

61. **Sadly Accepting**
```
sadly accepting expression with resigned peaceful eyes, small melancholic smile, features showing letting go, soft fading lighting, graceful surrender atmosphere, complex peace rendering
```

62. **Hopefully Uncertain**
```
hopefully uncertain expression with bright but questioning eyes, tentative hopeful smile, wanting to believe features, warm but wavering lighting, cautious optimism atmosphere, complex hope rendering
```

63. **Guilty but Defiant**
```
guilty but defiant expression with ashamed but stubborn eyes, set jaw despite downcast gaze, mix of shame and pride, harsh confrontational lighting, complex moral atmosphere, detailed conflict rendering
```

64. **Sweetly Mischievous**
```
sweetly mischievous expression with innocent wide eyes but knowing smile, angelic but impish combination, playful secret visible, warm playful lighting, charming troublemaker atmosphere, complex cute rendering
```

65. **Wearily Amused**
```
wearily amused expression with tired but smiling eyes, exhausted but entertained features, too-tired-to-care humor, soft dim lighting, end-of-rope humor atmosphere, complex tired humor rendering
```

66. **Protectively Angry**
```
protectively angry expression with fierce determined eyes, set protective jaw, mama-bear intensity, dramatic protective lighting, defending loved ones atmosphere, complex protective fury rendering
```

67. **Shyly Pleased**
```
shyly pleased expression with downcast but happy eyes, small embarrassed smile, modest delight visible, soft flattering lighting, humble happiness atmosphere, complex shy joy rendering
```

---

## ACTION TEMPLATES (143)

### Sports - Basketball (6)

1. **Dribbling Athletic**
```
dynamic basketball dribbling stance with ball bouncing at hand height, body low in athletic position with knees bent, focused concentrated expression watching defenders, indoor court environment with polished floor, dramatic sports lighting from above, detailed basketball texture and hand position, authentic basketball uniform
```

2. **Shooting Form**
```
basketball shooting form at release point with arm extended upward, ball leaving fingertips with perfect rotation, body elevated in jump shot, focused determined expression eyes on rim, gymnasium environment with hoop visible, dramatic action lighting freezing motion, detailed hand position and form, proper basketball attire
```

3. **Layup Drive**
```
dynamic layup drive with body elevated toward basket, ball extended in one hand toward hoop, opposite arm protecting, fierce determined expression, close to basket environment, intense action lighting, detailed athletic motion capture, game uniform with visible motion
```

4. **Defensive Stance**
```
intense defensive stance with arms spread wide and knees deeply bent, shuffling lateral movement suggested, focused tracking expression on opponent, court environment defensive position, athletic sports lighting, detailed defensive posture, proper basketball defensive attire
```

5. **Rebounding Jump**
```
powerful rebounding jump with arms stretched overhead reaching for ball, body at peak elevation, determined aggressive expression, under basket environment with backboard visible, dramatic action lighting, detailed jumping mechanics, athletic uniform showing effort
```

6. **Passing Motion**
```
basketball chest pass in motion with arms extending and ball releasing, body balanced with follow-through, focused accurate expression, team game environment, bright court lighting, detailed passing form and ball position, team uniform
```

### Sports - Soccer (5)

7. **Powerful Kick**
```
powerful soccer kick at moment of contact with full leg extension, planted foot stable, ball compression visible, intense focused expression, grass field environment with goal visible, dynamic outdoor lighting, detailed kicking mechanics, full soccer kit
```

8. **Dribbling Control**
```
skillful soccer dribbling with ball close to feet, body low with quick direction change, focused concentration on ball, pitch environment with defenders suggested, natural outdoor lighting, detailed ball control and footwork, proper soccer attire
```

9. **Header Jump**
```
soccer header at peak of jump with neck muscles engaged, eyes open tracking ball, body positioned for power or direction, aerial action environment, bright natural lighting, detailed heading technique, soccer uniform
```

10. **Goalkeeping Dive**
```
dramatic goalkeeper diving save with body fully extended horizontally, arms stretched toward ball, intense desperate expression, goal mouth environment, dramatic action lighting, detailed save technique, goalkeeper kit with gloves
```

11. **Tactical Slide Tackle**
```
committed slide tackle with body low on ground, leg extended to ball, focused determined expression, pitch environment grass flying, dynamic action lighting, detailed tackling form, dirty game-worn uniform
```

### Sports - Tennis (4)

12. **Forehand Power**
```
powerful tennis forehand at contact point with full body rotation, racket meeting ball with extension, focused aggressive expression, clay or hard court environment, natural outdoor lighting, detailed racket and ball contact, proper tennis attire
```

13. **Backhand Slice**
```
elegant tennis backhand slice with controlled racket motion, body balanced with smooth technique, concentrated precise expression, court environment with net visible, bright tennis lighting, detailed slice technique, classic tennis clothing
```

14. **Serve Toss**
```
tennis serve at peak of ball toss with arm extended high, body coiled ready to strike, focused upward gaze on ball, baseline serving position, dramatic lighting against sky, detailed serve preparation, traditional tennis whites possible
```

15. **Net Volley**
```
quick tennis volley at net with compact racket motion, body forward and reactive, alert responsive expression, close to net position, fast action lighting, detailed volley technique, tennis attire
```

### Sports - Swimming (4)

16. **Freestyle Stroke**
```
freestyle swimming mid-stroke with arm pulling through water, body streamlined with rotation, focused breathing expression, pool lane environment with water texture, underwater and surface lighting, detailed stroke technique, competitive swimwear and cap
```

17. **Backstroke Pull**
```
backstroke swimming with alternating arm motion, body flat on water surface, rhythmic steady expression, pool environment with lane lines, unique water lighting from below, detailed backstroke form, racing swimsuit
```

18. **Diving Entry**
```
competitive diving entry with body streamlined and pointed, arms extended overhead in tight position, focused controlled expression, pool diving environment, dramatic moment lighting, detailed entry form, competitive swimwear
```

19. **Breaststroke Glide**
```
breaststroke swimming in glide phase with arms extended forward, legs together after kick, peaceful powerful expression, pool environment, aquatic lighting through water, detailed breaststroke position, swimming attire
```

### Sports - Running/Track (5)

20. **Sprint Start**
```
explosive sprint start from blocks with body driving forward, arms pumping with power, fierce determined expression, track starting blocks environment, tense dramatic lighting, detailed starting mechanics, track uniform and spikes
```

21. **Mid-Sprint Full Speed**
```
full speed sprinting with body perfectly aligned, arms and legs in maximum drive, intense focused expression, track environment in motion blur, dynamic action lighting, detailed sprinting form, competition attire
```

22. **Victory Finish**
```
crossing finish line with arms raised in victory, body leaning through tape, triumphant elated expression, finish line environment with timer, celebratory bright lighting, detailed victory moment, race uniform with number
```

23. **Long Jump Flight**
```
long jump in flight phase with body extended forward, arms reaching with hitch kick legs, focused distance expression, sand pit approach environment, action capture lighting, detailed jump technique, track and field attire
```

24. **Hurdle Clear**
```
hurdling at clearance with lead leg extended, trail leg folding, focused forward expression, track hurdles environment, dynamic motion lighting, detailed hurdle technique, sprint attire
```

### Sports - Gymnastics (5)

25. **Perfect Handstand**
```
controlled handstand with body perfectly vertical, pointed toes and straight lines, focused balance expression, gymnastics mat environment, clean studio lighting, detailed balance and form, gymnastics leotard or attire
```

26. **Cartwheel Motion**
```
mid-cartwheel with body rotating sideways, hands on ground with legs splitting overhead, dynamic athletic expression, gymnasium floor environment, motion capture lighting, detailed cartwheel form, gymnastics attire
```

27. **Tucked Somersault**
```
tucked somersault rotation in mid-air, knees pulled tight to chest, controlled focused expression, tumbling environment, dynamic rotation lighting, detailed flip mechanics, gymnastics competition attire
```

28. **Split Leap**
```
split leap at peak height with legs in full horizontal split, arms gracefully extended, elegant powerful expression, dance studio or floor environment, dramatic performance lighting, detailed leap form, dance or gymnastics attire
```

29. **Balance Beam Pose**
```
elegant balance beam pose on one leg with arms artistically extended, other leg raised in attitude, poised confident expression, balance beam equipment environment, focused spotlight lighting, detailed balance form, competition leotard
```

### Sports - Martial Arts (6)

30. **High Kick**
```
powerful martial arts high kick with leg extended above head height, supporting leg stable, intense focused expression, dojo or training environment, dramatic martial arts lighting, detailed kick technique and form, traditional martial arts uniform
```

31. **Roundhouse Kick**
```
dynamic roundhouse kick with rotating hip and striking leg, arms positioned for balance, fierce determined expression, training space environment, action martial arts lighting, detailed spinning kick form, martial arts attire
```

32. **Defensive Block**
```
solid martial arts defensive block with arm raised to protect, body stable in ready stance, alert focused expression, dojo environment, practical training lighting, detailed blocking technique, traditional gi or training wear
```

33. **Punch Strike**
```
powerful martial arts punch at full extension, rotating hip generating power, intense focused expression, training or competition environment, dramatic action lighting, detailed striking form, martial arts attire
```

34. **Kata Stance**
```
formal kata stance demonstrating perfect form and control, traditional position with precise hand and foot placement, focused ceremonial expression, traditional dojo environment, respectful formal lighting, detailed traditional form, proper martial arts uniform
```

35. **Grappling Position**
```
martial arts grappling position showing control technique, body positioned for leverage, concentrated strategic expression, mat training environment, close action lighting, detailed grappling technique, grappling attire
```

### Sports - Additional (25)

36-39. **Volleyball** (serve, spike, set, dig)
40-43. **Baseball** (batting, pitching, catching, throwing)
44-46. **Golf** (drive, putt, chip)
47-49. **Skiing** (downhill, carving, jumping)
50-52. **Cycling** (road racing, mountain biking, sprint)
53-55. **Skateboarding** (kickflip, grinding, cruising)

### Fitness/Exercise (8)

56. **Push-Up Form**
```
push-up at bottom position with chest near floor, body in perfect plank alignment, arms bent at sides, core engaged with straight line from head to heels, focused determined expression, gym floor or mat environment, practical workout lighting, detailed push-up form and muscle engagement, athletic workout attire
```

57. **Squat Depth**
```
squat exercise at bottom depth with thighs parallel to floor, knees tracking over toes, back straight with chest up, arms forward for balance, determined effort expression, gym or home workout environment, clear functional lighting, detailed squat mechanics and alignment, athletic training clothes
```

58. **Plank Hold**
```
plank hold with body forming rigid straight line, forearms or hands on floor, core tightly engaged, legs extended with weight on toes, focused endurance expression, exercise mat environment, practical lighting, detailed plank alignment and form, athletic workout gear
```

59. **Deadlift Lockout**
```
deadlift at lockout standing tall with barbell at hip level, shoulders back and chest proud, hips fully extended, powerful satisfied expression, gym environment with equipment visible, strong gym lighting, detailed lifting form and posture, athletic training attire
```

60. **Bicep Curl**
```
bicep curl mid-repetition with dumbbell raised halfway, elbow fixed at side, muscle visibly contracting, focused concentration on exercise, gym environment with mirrors, bright gym lighting showing muscle definition, detailed curl form and engagement, sleeveless workout attire
```

61. **Jumping Jack Peak**
```
jumping jack at peak with arms overhead and legs spread wide, full body extension in cardio movement, energetic active expression, open exercise space, bright energetic lighting, detailed jumping form and coordination, athletic cardio attire
```

62. **Lunge Exercise**
```
lunging exercise with front leg bent at 90 degrees, back knee lowering toward floor, torso upright with arms at sides or on hips, focused balanced expression, gym or outdoor environment, clear functional lighting, detailed lunge alignment, athletic training wear
```

63. **Pull-Up Lift**
```
pull-up with chin above bar and engaged muscles, arms fully contracted, core stabilized, determined effort expression, gym bar equipment setting, athletic lighting from above, detailed pulling mechanics, athletic sleeveless top
```

### Yoga (5)

64. **Warrior One**
```
warrior one yoga pose with front knee deeply bent at 90 degrees over ankle, back leg straight and strong pressing heel down, arms raised overhead with palms together reaching high, steady focused gaze forward in drishti, peaceful studio environment with natural light streaming through windows, calming soft yoga lighting with warm tones, detailed muscular engagement and alignment visible, fitted yoga attire showing clean body lines
```

65. **Tree Balance**
```
tree pose balance with standing leg rooted firmly into ground, raised foot pressed against inner thigh of standing leg, arms in graceful prayer position at heart center or extended overhead like branches, peaceful meditative expression with soft focused eyes, serene yoga space environment with plants and natural elements, soft diffused natural lighting, detailed balance technique and alignment, comfortable breathable yoga clothing
```

66. **Downward Dog**
```
downward facing dog pose with body forming inverted V shape, hands pressing firmly into mat with fingers spread, heels reaching toward floor stretching calves, head relaxed between arms with neck long, focused controlled breathing expression, yoga studio environment with wooden floors, even ambient lighting, detailed alignment showing spine length and hip position, yoga practice attire
```

67. **Seated Meditation**
```
seated meditation pose in lotus position with legs crossed and feet on opposite thighs, hands resting on knees in mudra, spine tall and straight, eyes closed with peaceful serene expression, minimalist meditation space with cushion, soft ambient candlelight quality, detailed meditative stillness, loose comfortable meditation clothing
```

68. **Cobra Pose**
```
cobra yoga pose with chest lifted from floor, hands pressing into mat beside ribs, hips and legs grounded, gentle backbend with head tilted back slightly, calm controlled expression, yoga mat on floor environment, warm natural lighting, detailed backbend form and engagement, fitted yoga attire
```

### Dance (5)

69. **Ballet Arabesque**
```
ballet arabesque with one leg extended high behind body, supporting leg straight and strong, arms in graceful third position, elegant refined expression, ballet studio with barre and mirrors, classic dance lighting, detailed balletic line and form, traditional ballet attire with pointe shoes possible
```

70. **Contemporary Leap**
```
contemporary dance leap at peak height with legs in split, arms reaching expressively, body suspended in emotional movement, intense artistic expression, black box theater environment, dramatic stage lighting with colors, detailed contemporary dance form, flowing dance costume
```

71. **Hip Hop Freeze**
```
hip hop freeze pose in dramatic angular position, body creating geometric shapes, hands or head on ground for support, confident cool expression, urban dance studio environment, dynamic street dance lighting, detailed hip hop style and attitude, urban streetwear dance attire
```

72. **Salsa Partner Frame**
```
salsa partner dance frame with proper hold and connection, body in motion with hip movement suggested, engaging charming expression, dance floor environment with warm tones, latin club lighting with warm colors, detailed salsa posture and styling, latin dance attire
```

73. **Breakdance Power Move**
```
breakdance windmill on back with legs spinning in circular motion, hands planted for support and rotation, dynamic athletic expression, dance floor or cardboard surface, dramatic hip hop lighting, detailed power move mechanics, baggy breakdance attire
```

### Musical Performance (5)

74. **Guitar Playing**
```
guitar playing with fingers on fretboard forming chord, other hand strumming or picking strings, body connected to instrument, passionate musical expression, stage or studio environment, warm performance lighting with spotlight, detailed guitar playing technique, musician attire appropriate to genre
```

75. **Piano Performance**
```
piano playing with fingers dancing across keys, body slightly swaying with music, emotional connection to piece, absorbed musical expression, piano bench environment with instrument visible, soft performance lighting, detailed hand position on keys, concert or practice attire
```

76. **Drumming Energy**
```
drums playing with sticks raised in motion between beats, body keeping rhythm, energetic groove expression, drum kit environment with cymbals visible, dynamic stage lighting with movement, detailed drumming technique and energy, casual musician attire
```

77. **Singing Microphone**
```
singing into microphone with proper mic technique, body expressing through gesture, emotionally connected to lyrics, passionate performance expression, stage environment with audience suggestion, dramatic spotlight and stage lighting, detailed singing posture and engagement, performance outfit
```

78. **Violin Playing**
```
violin playing with bow drawing across strings, chin rest contact showing proper hold, body swaying with musical phrase, focused artistic expression, concert hall or studio environment, elegant performance lighting, detailed violin technique, formal concert attire
```

### Reading/Writing (4)

79. **Reading Absorbed**
```
reading book held comfortably in both hands with pages open at interesting angle, body in relaxed seated position in comfortable chair, completely absorbed expression with eyes scanning text, quiet cozy environment with bookshelves visible, warm lamp lighting creating reading atmosphere, detailed book and hand positioning showing engagement, casual comfortable loungewear clothing
```

80. **Writing Pen**
```
writing with pen in hand over paper or notebook, body leaning into task with focused concentration, thoughtful creative expression, desk or writing table environment, practical task lighting from lamp, detailed writing posture and pen grip, comfortable creative workspace attire
```

81. **Typing Laptop**
```
typing on laptop with fingers hovering over keyboard, screen glow illuminating face, engaged working expression, modern workspace environment, mixed screen and ambient lighting, detailed typing posture and hand position, professional casual work attire
```

82. **Studying Concentrated**
```
studying with head resting on hand over open textbook, concentrated learning expression, surrounded by study materials, library or desk environment, practical study lighting, detailed studying posture showing effort, student casual attire
```

### Eating/Drinking (4)

83. **Enjoying Meal**
```
eating meal with utensil raised bringing food toward mouth, enjoying delicious bite, satisfied pleased expression, dining table environment with meal visible, warm dining lighting, detailed eating posture and table manners, appropriate dining attire
```

84. **Drinking Glass**
```
drinking from glass or cup held elegantly, liquid tilting toward lips, refreshed enjoying expression, cafe or dining environment, warm ambient lighting, detailed drinking gesture and glass hold, casual or semiformal attire
```

85. **Coffee/Tea Comfort**
```
coffee or tea enjoyment holding warm mug with both hands, steam rising from beverage, contented peaceful expression, cozy cafe or home environment, warm morning lighting, detailed mug holding gesture, comfortable casual clothing
```

86. **Toast Celebration**
```
toast gesture raising wine glass with arm extended, celebratory cheers moment, happy social expression, celebration or dinner party environment, festive warm lighting, detailed toast posture and glass hold, elegant occasion attire
```

### Cooking (3)

87. **Stove Cooking**
```
cooking at stove stirring pot with wooden spoon, monitoring dish progress, focused culinary concentration, kitchen environment with equipment visible, warm practical kitchen lighting, detailed cooking action and tools, casual home attire with apron
```

88. **Chopping Vegetables**
```
chopping vegetables on cutting board with proper knife technique, focused precision in cuts, concentrated chef expression, kitchen prep area environment, clear task lighting, detailed knife skills and form, cooking attire with apron
```

89. **Tasting Dish**
```
tasting dish from spoon with evaluating expression, culinary assessment moment, thoughtful critical expression, active kitchen environment, warm cooking lighting, detailed tasting gesture, chef or home cooking attire
```

### Social Gestures (10)

90. **Friendly Wave**
```
friendly wave gesture with raised hand and palm facing outward, fingers spread in welcoming greeting motion, warm genuine smile on face, outdoor public space or doorway environment, natural daylight lighting, detailed wave gesture and inviting expression, casual everyday clothing appropriate for social situation
```

91. **Thumbs Up Approval**
```
enthusiastic thumbs up with arm extended and thumb raised prominently, encouraging approval gesture, bright positive smile showing support, casual social environment, warm positive lighting, detailed hand gesture showing enthusiasm, casual friendly attire
```

92. **Pointing Direction**
```
pointing gesture with arm extended and index finger indicating direction clearly, helpful guiding expression explaining something, contextual environment where direction matters, clear natural lighting, detailed pointing form and engaged expression, appropriate everyday clothing
```

93. **Shrugging Uncertain**
```
shrugging gesture with shoulders raised and palms turned upward, uncertain or questioning expression, casual conversational body language, everyday environment, natural ambient lighting, detailed shrug posture and expression, casual clothing
```

94. **Clapping Applause**
```
clapping celebration with hands meeting in applause, genuine appreciation or congratulations, happy proud expression, event or performance environment, warm celebration lighting, detailed clapping motion and joyful expression, occasion-appropriate attire
```

95. **Handshake Greeting**
```
handshake greeting with firm professional grip, direct eye contact showing respect, confident professional expression, business or social introduction environment, professional lighting, detailed handshake form and engaged expression, professional or semiformal attire
```

96. **Hugging Embrace**
```
hugging embrace with arms wrapped around another in warm gesture, comforting supportive body language, caring emotional expression, intimate reunion or comfort environment, soft emotional lighting, detailed embrace showing genuine connection, casual everyday clothing
```

97. **Thinking Pose**
```
thinking pose with hand on chin in contemplation, considering options or remembering, thoughtful processing expression, quiet environment for reflection, calm ambient lighting, detailed thinking gesture and expression, comfortable casual attire
```

98. **Excited Cheering**
```
excited cheering with arms raised in victory or celebration, explosive happy energy, overjoyed ecstatic expression, celebratory event environment, bright festive lighting, detailed celebration gesture and emotion, fun casual or event attire
```

99. **Comforting Touch**
```
comforting pat on shoulder reaching to console, supportive caring gesture, sympathetic kind expression, private supportive moment environment, soft warm lighting, detailed comforting touch and expression, everyday comfortable clothing
```

### Daily Activities (10)

100. **Morning Stretch**
```
waking up stretching in bed with arms raised overhead, morning energy release, sleepy awakening expression, bedroom environment with morning light, soft golden morning lighting through window, detailed stretching gesture, comfortable sleepwear or pajamas
```

101. **Brushing Teeth**
```
brushing teeth at bathroom sink with toothbrush in mouth, morning hygiene routine, sleepy routine expression, bathroom environment with mirror visible, bright bathroom lighting, detailed tooth brushing action, casual morning attire or pajamas
```

102. **Getting Dressed**
```
getting dressed pulling shirt on or adjusting clothing, daily preparation moment, neutral focused expression, bedroom or closet environment, natural room lighting, detailed dressing action, partially dressed showing process
```

103. **Casual Walking**
```
walking casually in relaxed stride with natural arm swing, everyday locomotion, calm content expression, sidewalk or hallway environment, natural ambient lighting, detailed walking gait, casual everyday clothing
```

104. **Opening Door**
```
opening door with hand on handle turning or pulling, transitional everyday action, neutral purposeful expression, doorway environment, practical lighting, detailed door opening gesture, everyday clothing
```

105. **Sitting Down**
```
sitting down onto chair or couch in process of lowering, transitional seated movement, relaxed comfortable expression, living space environment, warm home lighting, detailed sitting action, casual home attire
```

106. **Phone Checking**
```
looking at phone with device held at viewing angle, modern daily activity, engaged or curious expression, any casual environment, mixed screen and ambient lighting, detailed phone viewing posture, contemporary casual clothing
```

107. **Carrying Bag**
```
carrying bag or backpack slung over shoulder, everyday transport, purposeful walking expression, urban or school environment, natural outdoor or indoor lighting, detailed carrying posture, appropriate casual attire
```

108. **Cleaning/Tidying**
```
cleaning or tidying with cloth or tool in hand, household task action, focused productive expression, home environment, practical daylight or room lighting, detailed cleaning action, casual work-appropriate home clothing
```

109. **Pet Greeting**
```
greeting pet with affectionate bend toward animal, loving interaction moment, warm happy expression, home environment, soft warm lighting, detailed pet interaction posture, comfortable home attire
```

### Team Activities (6)

110-115. Team huddle, high five, group discussion, collaborative work, team celebration, passing coordination

### Climbing/Adventure (4)

116-119. Rock climbing, hiking trail, zip lining, surfing wave

### Interaction Gestures (10)

120-129. Offering help, receiving gift, showing object, asking question, nodding agreement, shaking head no, beckoning gesture, pushing away, leaning to listen, stepping back surprised

### Extreme Sports (4)

130-133. Parkour vault, snowboarding carve, bungee jumping, motocross jump

### Water Activities (3)

134-136. Kayaking paddle, scuba diving swim, water polo treading

### Creative Activities (4)

137-140. Painting canvas, photography composing, sculpting clay, drawing sketchbook

### Communication/Presentation (3)

141-143. Public speaking, teaching whiteboard, interview conversation

---

## Style Variations

Use these style prefixes with templates:

1. `3d animation, pixar style, high quality, detailed`
2. `3d animated character, smooth shading, studio lighting`
3. `pixar-style 3d render, professional quality`
4. `3d cg animation, clean render`
5. `high-quality 3d animation, detailed modeling`
6. `pixar style 3d animation, soft lighting`
7. `pixar style 3d animation, smooth shading, PBR materials, subsurface scattering on skin, soft ambient occlusion, professional CGI render`

## Quality Suffix

Add to end of prompts for consistent quality:
```
pixar style 3d animated character
```

---

## Quick Reference - Popular Templates

### Most Versatile Poses
- Neutral Standing (#2)
- Casual Lean (#5)
- Relaxed Seated (#16)
- Walking Natural (#31)
- Three-Quarter Dynamic (#42)

### Most Expressive Emotions
- Genuine Warm Smile (#1)
- Deeply Sorrowful (#11)
- Completely Shocked (#27)
- Bittersweet Smile (#50)
- Determined Despite Fear (#55)

### Most Dynamic Actions
- Running Athletic (#32)
- Jumping Joy (#33)
- Basketball Shooting (#2)
- Martial Arts High Kick (#30)
- Ballet Arabesque (#69)

---

*Document generated: 2025-12-06*
*Total templates: 276 (66 Pose + 67 Expression + 143 Action)*
*Source: vocabulary_templates_v2_detailed.py*
