"""
Necker Cube Superposition Experiment v3
========================================

Updates:
- Auto-saves after each trial (crash-proof)
- "Ready" button when you've found superposition
- "Bad trial" option (press 0)
- Shorter trials (15-20s)
- Fewer trials (5 per condition = 20 total)

Run: python necker_experiment_v3.py
"""

import pygame
import random
import time
import csv
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

TRIALS_PER_CONDITION = 5   # 5 each = 20 total (~10-15 min)
STATE_DURATION = (15, 20)  # seconds (shorter)
TREE_ONSET = (8, 15)       # seconds into trial
TREE_DURATION = 0.4        # seconds

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BG_COLOR = (128, 128, 128)
CUBE_COLOR = (0, 0, 0)
CUBE_LINE_WIDTH = 3

OUTPUT_DIR = Path.home() / "Downloads" / "necker_experiment"

# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def draw_necker_cube(screen, center_x, center_y, size=200):
    s = size
    offset = size * 0.4
    
    front = [
        (center_x - s//2, center_y - s//2),
        (center_x + s//2, center_y - s//2),
        (center_x + s//2, center_y + s//2),
        (center_x - s//2, center_y + s//2),
    ]
    
    back = [
        (center_x - s//2 + offset, center_y - s//2 - offset),
        (center_x + s//2 + offset, center_y - s//2 - offset),
        (center_x + s//2 + offset, center_y + s//2 - offset),
        (center_x - s//2 + offset, center_y + s//2 - offset),
    ]
    
    pygame.draw.lines(screen, CUBE_COLOR, True, front, CUBE_LINE_WIDTH)
    pygame.draw.lines(screen, CUBE_COLOR, True, back, CUBE_LINE_WIDTH)
    
    for i in range(4):
        pygame.draw.line(screen, CUBE_COLOR, front[i], back[i], CUBE_LINE_WIDTH)


def draw_tree(screen, center_x, center_y, size=150):
    trunk_width = size // 5
    trunk_height = size // 2
    trunk_rect = pygame.Rect(
        center_x - trunk_width // 2,
        center_y,
        trunk_width,
        trunk_height
    )
    pygame.draw.rect(screen, (101, 67, 33), trunk_rect)
    
    foliage = [
        (center_x, center_y - size // 2),
        (center_x - size // 2, center_y + size//4),
        (center_x + size // 2, center_y + size//4),
    ]
    pygame.draw.polygon(screen, (34, 139, 34), foliage)


def draw_fixation(screen, center_x, center_y, size=20):
    pygame.draw.line(screen, (0, 0, 0), 
                    (center_x - size, center_y), 
                    (center_x + size, center_y), 3)
    pygame.draw.line(screen, (0, 0, 0), 
                    (center_x, center_y - size), 
                    (center_x, center_y + size), 3)


def draw_text(screen, text, x, y, font, color=(0, 0, 0)):
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(x, y))
    screen.blit(surface, rect)

# =============================================================================
# SAVE FUNCTION (crash-proof)
# =============================================================================

def save_results(results, output_file):
    """Save results to CSV (called after each trial)"""
    if not results:
        return
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Necker Cube Experiment v3")
    clock = pygame.time.Clock()
    
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 28)
    
    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create trial list
    conditions = ['LEFT', 'RIGHT', 'BOTH', 'CONTROL']
    trials = []
    for cond in conditions:
        for _ in range(TRIALS_PER_CONDITION):
            trials.append({
                'condition': cond,
                'state_duration': random.uniform(*STATE_DURATION),
                'tree_onset': random.uniform(*TREE_ONSET),
            })
    random.shuffle(trials)
    
    results = []
    
    session_start = datetime.now()
    session_id = session_start.strftime("%Y-%m-%d--%H-%M-%S")
    output_file = OUTPUT_DIR / f"necker_results_{session_id}.csv"
    
    # Instructions
    screen.fill(BG_COLOR)
    instructions = [
        "NECKER CUBE EXPERIMENT v3",
        "",
        "LEFT: Hold front face on LEFT",
        "RIGHT: Hold front face on RIGHT", 
        "BOTH: Hold BOTH interpretations (superposition)",
        "CONTROL: Just watch fixation cross",
        "",
        "NEW: Press SPACE when you've FOUND IT",
        "     Then the trial timer starts",
        "",
        "After tree: press 1-5 for response",
        "Press 0 if trial was BAD (distracted, lost it, etc)",
        "",
        f"20 trials total. Saves after each trial.",
        "",
        "Press SPACE to start",
    ]
    
    for i, line in enumerate(instructions):
        draw_text(screen, line, center_x, 80 + i * 38, font_medium)
    
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
    
    # Run trials
    for trial_num, trial in enumerate(trials):
        
        condition = trial['condition']
        state_duration = trial['state_duration']
        tree_onset = min(trial['tree_onset'], state_duration - 2)
        
        # --- Instruction screen ---
        screen.fill(BG_COLOR)
        draw_text(screen, f"Trial {trial_num + 1} / {len(trials)}", center_x, 100, font_large)
        draw_text(screen, f"Condition: {condition}", center_x, 180, font_large)
        
        if condition == 'LEFT':
            draw_text(screen, "Hold front face on LEFT", center_x, 280, font_medium)
        elif condition == 'RIGHT':
            draw_text(screen, "Hold front face on RIGHT", center_x, 280, font_medium)
        elif condition == 'BOTH':
            draw_text(screen, "Hold BOTH (superposition)", center_x, 280, font_medium)
        else:
            draw_text(screen, "Just relax, watch the cross", center_x, 280, font_medium)
        
        draw_text(screen, "Press SPACE to see the cube", center_x, 450, font_small)
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
        
        # --- Search phase: show cube, wait for SPACE when ready ---
        search_start = datetime.now()
        
        if condition != 'CONTROL':
            # Show cube, wait for "I found it"
            finding = True
            while finding:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            finding = False
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                
                screen.fill(BG_COLOR)
                draw_necker_cube(screen, center_x, center_y)
                draw_text(screen, "Find it... press SPACE when locked in", center_x, 700, font_small, (80, 80, 80))
                pygame.display.flip()
                clock.tick(60)
        
        search_end = datetime.now()
        search_duration = (search_end - search_start).total_seconds()
        
        # --- Fixation flash (0.5 sec) ---
        screen.fill(BG_COLOR)
        if condition == 'CONTROL':
            draw_fixation(screen, center_x, center_y)
        else:
            draw_necker_cube(screen, center_x, center_y)
        pygame.display.flip()
        time.sleep(0.5)
        
        # --- State phase (timed) ---
        trial_start = datetime.now()
        tree_shown = False
        tree_start_time = None
        tree_end_time = None
        
        state_start = time.time()
        
        while True:
            current_time = time.time() - state_start
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
            
            screen.fill(BG_COLOR)
            
            showing_tree = False
            if current_time >= tree_onset and current_time < tree_onset + TREE_DURATION:
                showing_tree = True
                if not tree_shown:
                    tree_start_time = datetime.now()
                    tree_shown = True
            
            if showing_tree:
                draw_tree(screen, center_x, center_y)
                if tree_end_time is None and current_time >= tree_onset + TREE_DURATION:
                    tree_end_time = datetime.now()
            elif condition == 'CONTROL':
                draw_fixation(screen, center_x, center_y)
            else:
                draw_necker_cube(screen, center_x, center_y)
            
            remaining = max(0, state_duration - current_time)
            timer_text = f"{remaining:.0f}s"
            timer_surface = font_small.render(timer_text, True, (100, 100, 100))
            screen.blit(timer_surface, (SCREEN_WIDTH - 60, 20))
            
            pygame.display.flip()
            clock.tick(60)
            
            if current_time >= state_duration:
                break
        
        if tree_end_time is None:
            tree_end_time = datetime.now()
        
        trial_end = datetime.now()
        
        # --- Response screen ---
        screen.fill(BG_COLOR)
        draw_text(screen, "What happened after the tree?", center_x, 100, font_large)
        draw_text(screen, "Press 0 if this trial was BAD", center_x, 150, font_small, (150, 50, 50))
        
        if condition == 'BOTH':
            options = [
                "1: SURVIVED (never broke)",
                "2: BROKE -> snapped LEFT (clean)",
                "3: BROKE -> snapped RIGHT (clean)",
                "4: BROKE -> WOBBLED (flickered between)",
                "5: REFORMED -> snapped back to both",
                "6: REFORMED -> wobbled back to both",
                "7: Unsure",
            ]
        elif condition in ['LEFT', 'RIGHT']:
            options = [
                "1: HELD steady (no change)",
                "2: FLIPPED (clean snap to other)",
                "3: WOBBLED (flickered)",
                "4: Went to BOTH (superposition)",
                "5: Disrupted -> snapped back",
                "6: Disrupted -> wobbled back",
                "7: Unsure",
            ]
        else:
            options = [
                "1: Nothing happened",
                "2: Tree was distracting",
                "3: Other",
            ]
        
        for i, opt in enumerate(options):
            draw_text(screen, opt, center_x, 250 + i * 50, font_medium)
        
        pygame.display.flip()
        
        response = None
        response_time = None
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7]:
                        response = int(event.unicode)
                        response_time = datetime.now()
                        waiting = False
        
        # Store result
        result = {
            'trial_num': trial_num + 1,
            'condition': condition,
            'search_duration': search_duration,
            'trial_start': trial_start.isoformat(),
            'tree_onset_planned': tree_onset,
            'tree_start': tree_start_time.isoformat() if tree_start_time else '',
            'tree_end': tree_end_time.isoformat() if tree_end_time else '',
            'trial_end': trial_end.isoformat(),
            'response': response,
            'bad_trial': response == 0,
            'response_time': response_time.isoformat() if response_time else '',
            'state_duration': state_duration,
        }
        results.append(result)
        
        # SAVE AFTER EACH TRIAL (crash-proof)
        save_results(results, output_file)
        
        # Brief pause
        screen.fill(BG_COLOR)
        status = "MARKED BAD" if response == 0 else "Recorded"
        draw_text(screen, f"{status}. Next trial in 2s...", center_x, center_y, font_medium)
        draw_text(screen, f"Saved to: {output_file.name}", center_x, center_y + 50, font_small, (100, 100, 100))
        pygame.display.flip()
        time.sleep(2.0)
    
    # --- End screen ---
    screen.fill(BG_COLOR)
    draw_text(screen, "EXPERIMENT COMPLETE", center_x, 200, font_large)
    
    good_trials = sum(1 for r in results if not r['bad_trial'])
    draw_text(screen, f"{good_trials} good trials, {len(results) - good_trials} marked bad", center_x, 300, font_medium)
    draw_text(screen, f"Saved: {output_file}", center_x, 400, font_small)
    draw_text(screen, "Press SPACE to exit", center_x, 550, font_small)
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_SPACE, pygame.K_ESCAPE]:
                    waiting = False
    
    print(f"\nResults saved to: {output_file}")
    pygame.quit()


if __name__ == "__main__":
    run_experiment()
