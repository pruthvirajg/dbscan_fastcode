#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>

#include "../include/config.h"
#include "../include/utils.h"
#include "../include/dbscan.h"

void print_dataset(){
   for(int i=0; i< OBSERVATIONS; i++){
      printf("%d, %s, ", i, dataset[i].name);
      
      for(int j=0; j< FEATURES; j++){
         printf("%d, ", dataset[i].features[j]);
      }

      printf("%d, ", dataset[i].class);
      printf("%d", dataset[i].label);
      printf("\n");
   }

}

void load_dataset(){
   FILE *fp;
   char buffer[10000];
   char *pbuff;
   char *ch;
   char delim[] = ",";

   dataset = (dataset_t *) malloc(sizeof(dataset_t)*OBSERVATIONS);

   fp = fopen(FILE_PATH, "r");

   int struct_counter = 0;
   int observation_count = 0;

   while (1) {
      if (!fgets(buffer, sizeof buffer, fp)) break;
      pbuff = buffer;

      ch = strtok(pbuff, delim);

      while (ch != NULL) {
         if(struct_counter == 0){
            dataset[observation_count].name = (char *)malloc(sizeof(char)*strlen(ch));
            strcpy(dataset[observation_count].name, ch);
         }
         else if(struct_counter>=1 && struct_counter <= FEATURES){
            dataset[observation_count].features[struct_counter - 1] = atoi(ch);
         }
         else if(struct_counter == FEATURES + 1){
            dataset[observation_count].class = atoi(ch);
         }
         else if(struct_counter == FEATURES + 2){
            dataset[observation_count].label = atoi(ch);
         }
         else{
            assert(struct_counter <= FEATURES + 2);
         }

         ch = strtok(NULL, delim);
         struct_counter++;
      }

      observation_count++;
      struct_counter = 0;
   }
   
   fclose(fp);

}

void free_dataset(){
   for(int i=0; i< OBSERVATIONS; i++){
      free(dataset[i].name);
   }

   free(dataset);

}
