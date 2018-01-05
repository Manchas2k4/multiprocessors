#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "list.h"

struct node* create_node(unsigned int val, struct node *nxt) {
    struct node* n;
    n = (struct node*) malloc( sizeof(struct node) );
    if (n == NULL) {
        return NULL;
    }
    n->value = val;
    n->next = nxt;
    return n;
}

struct list* create_list() {
	struct list *lst;
	
	lst = (struct list*) malloc (sizeof(struct list));
	if (lst == NULL) {
		return NULL;
	}
	lst->head = NULL;
	lst->size = 0;
	return lst;
}

int add_first(struct list *lst, unsigned int val) {
	struct node *n;
	
	n = create_node(val, NULL);
	if (n == NULL) {
		return 0;
	}
	n->next = lst->head;
	lst->head = n;
	lst->size++;
	return 1;
}

int add_last(struct list *lst, unsigned int val) {
    struct node *n, *p;
    
    n = create_node(val, NULL);
	if (n == NULL) {
		return 0;
	}
	if (is_empty(lst)) {
		return add_first(lst, val);
	} 
	
	p = lst->head;
	while (p->next != NULL) {
		p = p->next;
	}
	
	n->next = p->next;
	p->next = n;
	lst->size++;
	return 1;
}

int add_before(struct list *lst, unsigned int val, unsigned int ele) {
	struct node *n, *p, *q;
    
    n = create_node(val, NULL);
	if (n == NULL) {
		return 0;
	}
	
	p = lst->head;
	q = NULL;
	while (p != NULL && p->value != ele) {
		q = p;
		p = p->next;
	}
	
	if (q == NULL) {
		return add_first(lst, val);
	}
	
	if (p == NULL) {
		n->next = q->next;
		q->next = n;
		lst->size++;
		return 1;
	}
	
	q->next = n;
	n->next = p;
	lst->size++;
    return 1;
}

int remove_first(struct list *lst) {
	struct node *p;
	unsigned int val;
	
    if (is_empty(lst)) {
    	return -1;
    }
    
    p = lst->head;
    lst->head = p->next;
    lst->size--;
    
    val = p->value;
    free(p);
    return val;
}

int remove_last(struct list *lst) {
    struct node *p, *q;
	unsigned int val;
	
    if (is_empty(lst)) {
    	return -1;
    }
    
    p = lst->head;
    q = NULL;
    while (p->next != NULL) {
    	q = p;
    	p = p->next;
    }
    
    if (q == NULL) {
    	return remove_first(lst);
    }
    
    q->next = p->next;
    lst->size--;
    
    val = p->value;
    free(p);
    return val;
}

int remove_value(struct list *lst, unsigned int ele) {
    struct node *p, *q;
	unsigned int val;
	
    if (is_empty(lst)) {
    	return 0;
    }
    
    p = lst->head;
    q = NULL;
    while (p != NULL && p->value != ele) {
    	q = p;
    	p = p->next;
    }
    
    if (q == NULL) {
    	remove_first(lst);
    	return 1;
    }
    
    if (p != NULL) {
    	q->next = p->next;
    	lst->size--;
    	free(p);
    	return 1;
    }
    
    return 0;
}

void display_list(struct list *lst) {
    struct node *p;
    
    p = lst->head;
    while (p != NULL) {
    	printf("%i ", p->value);
    	p = p->next;
    }
    printf("\n");
}

void destroy_list(struct list *lst) {
    remove_all(lst);
}

unsigned int length(struct list *lst) {
    return lst->size;
}

int find_value(struct list *lst, unsigned int ele) {
    struct node *p;
    
    p = lst->head;
    while (p != NULL) {
    	if (p->value == ele) {
    		return 1;
    	}
    	p = p->next;
    }
    return 0;
}

int is_empty(struct list *lst) {
    return (lst->head == NULL);
}

void remove_all(struct list *lst) {
    struct node *p, *q;
    
    p = lst->head;
    while (p != NULL) {
    	q = p->next;
    	free(p);
    	p = q;
    }
    lst->head = NULL;
    lst->size = 0;
}


