/**
 * @file queue.h
 * @brief Header file for implementing queue operations.
 *
 * Assignment for basic C skills diagnostic.
 * Developed for courses 15-213/18-213/15-513 by R. E. Bryant, 2017
 * Extended to store strings, 2018
 *
 * @author Ashwin Venkatram <ashwinve@andrew.cmu.edu>
 */

#include <stdbool.h>
#include "config.h"

/************** Data structure declarations ****************/

/**
 * @brief Linked list element containing a string.
 *
 * You shouldn't change this struct.
 */
typedef struct list_ele {
    /**
     * @brief Pointer to a char array containing a string value.
     *
     * The memory for this string should be explicitly allocated and freed
     * whenever an element is inserted and removed from the queue.
     */
    DTYPE_OBS row_index;

    /**
     * @brief Pointer to the next element in the linked list.
     */
    struct list_ele *next;
} list_ele_t;

/**
 * @brief Queue structure representing a list of elements
 */
typedef struct {
    /**
     * @brief Pointer to the first element in the queue, or NULL if the
     *        queue is empty.
     */
    list_ele_t *head;
    /* pointer to last element in the queue; trivially NULL if the
       queue is empty, tail==head when 1 element in the queue
    */
    list_ele_t *tail;

    /*
     * TODO: You will need to add more fields to this structure
     *       to efficiently implement q_size and q_insert_tail
     */
    int qSize; // stores index of row

} queue_t;

/************** Operations on queue ************************/

/* Create empty queue. */
queue_t *queue_new(void);

/* Free ALL storage used by queue. */
void queue_free(queue_t *q);

/* Attempt to insert element at head of queue. */
bool queue_insert_head(queue_t *q, DTYPE_OBS row_index);

/* Attempt to insert element at tail of queue. */
bool queue_insert_tail(queue_t *q, DTYPE_OBS row_index);

/* Attempt to remove element from head of queue. */
bool queue_remove_head(queue_t *q);

/* Return number of elements in queue. */
size_t queue_size(queue_t *q);

/* Reverse elements in queue */
void queue_reverse(queue_t *q);
