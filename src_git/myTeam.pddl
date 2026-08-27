;Header and description

(define (domain pacman_enhanced)

    (:requirements :strips :typing :negative-preconditions)
    
    (:types 
        enemy team - object
        enemy1 enemy2 - enemy
        ally current_agent - team
    )

    (:predicates 
        ;Basic predicates
        (enemy_around ?e - enemy ?a - team)
        (is_pacman ?x)
        (food_in_backpack ?a - team)
        (food_available)
        (few_food_left)
        
        ;Goal predicates
        (defend_foods)
        
        ;Advanced predicates
        (enemy_long_distance ?e - enemy ?a - current_agent)
        (enemy_medium_distance ?e - enemy ?a - current_agent)
        (enemy_short_distance ?e - enemy ?a - current_agent)
        
        (3_food_in_backpack ?a - team)
        (5_food_in_backpack ?a - team)
        (10_food_in_backpack ?a - team)
        (20_food_in_backpack ?a - team)
        
        (near_food ?a - current_agent)
        (near_capsule ?a - current_agent)
        (capsule_available)
        
        (winning_gt3)
        (winning_gt5)
        (winning_gt10)
        (winning_gt20)
        
        (near_ally ?a - current_agent)
        (is_scared ?x)
    )

    ;; Action: Attack enemy territory to collect food
    (:action attack
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and 
            (not (is_pacman ?e1))
            (not (is_pacman ?e2))
            (food_available)
            (not (5_food_in_backpack ?a))
        )
        :effect (and 
            (not (food_available))
        )
    )

    ;; Action: Defend against invaders
    (:action defence
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and 
            (is_pacman ?e)
            (not (is_pacman ?a))
        )
        :effect (and 
            (not (is_pacman ?e))
        )
    )

    ;; Action: Return home to deposit food
    (:action go_home
        :parameters (?a - current_agent)
        :precondition (and 
            (is_pacman ?a)
            (food_in_backpack ?a)
        )
        :effect (and 
            (not (is_pacman ?a))
            (not (food_in_backpack ?a))
        )
    )

    ;; Action: Patrol when winning
    (:action patrol
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and 
            (not (is_pacman ?a))
            (not (is_pacman ?e1))
            (not (is_pacman ?e2))
            (winning_gt10)
        )
        :effect (and 
            (defend_foods)
        )
    )
    
    ;; Action: Eat capsule for power
    (:action eat_capsule
        :parameters (?a - current_agent)
        :precondition (and 
            (near_capsule ?a)
            (capsule_available)
        )
        :effect (and 
            (not (capsule_available))
        )
    )

)