/**
 * @file queue.c
 * @brief Implementation of a queue that supports FIFO and LIFO operations.
 *
 * This queue implementation uses a singly-linked list to represent the
 * queue elements. Each queue element stores a string value.
 *
 * Assignment for basic C skills diagnostic.
 * Developed for courses 15-213/18-213/15-513 by R. E. Bryant, 2017
 * Extended to store strings, 2018
 *
 * @author Ashwin Venkatram <ashwinve@andrew.cmu.edu>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/queue.h"

/**
 * @brief Allocates a new queue
 * @return The new queue, or NULL if memory allocation failed
 */
queue_t *queue_new(void) {
    queue_t *q = malloc(sizeof(queue_t));
    /* What if malloc returned NULL? */

    // checking for NULL queue
    if (q == NULL) {
        return NULL;
    }

    q->head = NULL;
    q->tail = NULL;
    q->row_index = 0;

    return q;
}

/**
 * @brief Frees all memory used by a queue
 * @param[in] q The queue to free
 */
void queue_free(queue_t *q) {
    if (q != NULL) {
        if (q->row_index > 0) {
            /* How about freeing the list elements and the strings? */
            list_ele_t *current = q->head;
            list_ele_t *nextVal;

            while (current != NULL) {
                nextVal = current->next;
                free(current->value);
                free(current);
                current = nextVal;
            }
        }

        free(q);
    }
}

/**
 * @brief Attempts to insert an element at head of a queue
 *
 * This function explicitly allocates space to create a copy of `s`.
 * The inserted element points to a copy of `s`, instead of `s` itself.
 *
 * @param[in] q The queue to insert into
 * @param[in] s String to be copied and inserted into the queue
 *
 * @return true if insertion was successful
 * @return false if q is NULL, or memory allocation failed
 */
bool queue_insert_head(queue_t *q, const char *s) {
    list_ele_t *newh;

    /* What should you do if the q is NULL? */
    if (q == NULL) {
        return false;
    } else {

        newh = malloc(sizeof(list_ele_t));

        /* Don't forget to allocate space for the string and copy it */
        /* What if either call to malloc returns NULL? */
        if (newh == NULL) {
            return false;

        } else {
            newh->value = malloc((strlen(s) + 1));

            if (newh->value == NULL) {
                free(newh);
                return false;
            } else {
                memset(newh->value, '\0', strlen(s) + 1);
                strcpy(newh->value, s);

                newh->next = q->head;
                q->head = newh;

                // incrementing the size
                q->row_index += 1;

                // update tail pointer when first element is added
                if (q->row_index == 1) {
                    q->tail = newh;
                }
            }
        }
    }
    return true;
}

/**
 * @brief Attempts to insert an element at tail of a queue
 *
 * This function explicitly allocates space to create a copy of `s`.
 * The inserted element points to a copy of `s`, instead of `s` itself.
 *
 * @param[in] q The queue to insert into
 * @param[in] s String to be copied and inserted into the queue
 *
 * @return true if insertion was successful
 * @return false if q is NULL, or memory allocation failed
 */
bool queue_insert_tail(queue_t *q, const char *s) {
    /* You need to write the complete code for this function */
    /* Remember: It should operate in O(1) time */

    list_ele_t *tailVal;

    if (q == NULL) {
        return false;
    } else {
        tailVal = malloc(sizeof(list_ele_t));
        if (tailVal == NULL) {
            return false;
        } else {
            tailVal->value = malloc((strlen(s) + 1));

            if (q->row_index == 0) {
                if (tailVal->value == NULL) {
                    free(tailVal);
                    return false;
                } else {
                    memset(tailVal->value, '\0', strlen(s) + 1);
                    strcpy(tailVal->value, s);
                    tailVal->next = NULL;
                    // tail itself in NULL in the base case, needs assignment
                    q->tail = tailVal;
                }
            } else {
                if (!tailVal->value) {
                    free(tailVal);
                    return false;
                } else {
                    memset(tailVal->value, '\0', strlen(s) + 1);

                    strcpy(tailVal->value, s);

                    tailVal->next = NULL;

                    // assigning ->next of currently last element to point to
                    // new value
                    q->tail->next = tailVal;
                    q->tail = tailVal;
                }
            }

            // incrementing the size
            q->row_index += 1;

            // update head pointer when first element is added
            if (q->row_index == 1) {
                q->head = tailVal;
            }

            return true;
        }
    }
}

/**
 * @brief Attempts to remove an element from head of a queue
 *
 * If removal succeeds, this function frees all memory used by the
 * removed list element and its string value before returning.
 *
 * If removal succeeds and `buf` is non-NULL, this function copies up to
 * `bufsize - 1` characters from the removed string into `buf`, and writes
 * a null terminator '\0' after the copied string.
 *
 * @param[in]  q       The queue to remove from
 * @param[out] buf     Output buffer to write a string value into
 * @param[in]  bufsize Size of the buffer `buf` points to
 *
 * @return true if removal succeeded
 * @return false if q is NULL or empty
 */
bool queue_remove_head(queue_t *q, char *buf, size_t bufsize) {
    /* You need to fix up this code. */
    if (q == NULL || q->row_index == 0) {
        return false;
    }

    list_ele_t *current;
    current = q->head;
    q->head = q->head->next;

    if (bufsize > 0 && buf != NULL) {
        // copies n elements into buffer
        strncpy(buf, current->value, bufsize);
        buf[bufsize - 1] = '\0';
    }

    q->row_index -= 1;

    // handle the case where entire queue is deleted by tail points to
    // irrelevant memory location
    if (q->row_index == 0) {
        q->tail = NULL;
    }

    free(current->value);
    free(current);

    return true;
}

/**
 * @brief Returns the number of elements in a queue
 *
 * This function runs in O(1) time.
 *
 * @param[in] q The queue to examine
 *
 * @return the number of elements in the queue, or
 *         0 if q is NULL or empty
 */
size_t queue_size(queue_t *q) {
    /* You need to write the code for this function */
    /* Remember: It should operate in O(1) time */
    if (q == NULL || q->row_index == 0) {
        return 0;
    } else {
        return (size_t)q->row_index;
    }
}

/**
 * @brief Reverse the elements in a queue
 *
 * This function does not allocate or free any list elements, i.e. it does
 * not call malloc or free, including inside helper functions. Instead, it
 * rearranges the existing elements of the queue.
 *
 * @param[in] q The queue to reverse
 */
void queue_reverse(queue_t *q) {
    /* You need to write the code for this function */
    list_ele_t *prevElement;
    list_ele_t *currElement;
    list_ele_t *nextElement;

    if (q != NULL && q->row_index > 0) {
        prevElement = NULL;
        currElement = q->head;
        nextElement = currElement->next;

        // trivially first element now becomes the tail
        q->tail = currElement;

        while (currElement != NULL) {
            nextElement = currElement->next;

            currElement->next = prevElement;

            prevElement = currElement;
            currElement = nextElement;
        }

        // trivially upon currElement== NULL, prevElement is the head
        q->head = prevElement;
    }
}
