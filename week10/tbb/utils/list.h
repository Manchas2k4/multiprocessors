#ifndef LIST_H
#define LIST_H

typedef struct node {
    unsigned int value;
    struct node *next;
} Node;

typedef struct list {
    unsigned int size;
    struct node *head;
} List;

struct list* create_list();
int add_first(struct list*, unsigned int);
int add_last(struct list*, unsigned int);
int add_before(struct list*, unsigned int, unsigned int);
int remove_first(struct list*);
int remove_last(struct list*);
int remove_value(struct list*, unsigned int);
void display_list(struct list*);
unsigned int length(struct list*);
int find_value(struct list*, unsigned int);
int is_empty(struct list*);
void remove_all(struct list*);
void destroy_list(struct list*);

#endif
