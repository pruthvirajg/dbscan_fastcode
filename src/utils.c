#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include <sys/stat.h>   // stat
#include <stdbool.h>    // bool type

#include "../include/config.h"
#include "../include/utils.h"
#include "../include/dbscan.h"


bool file_exists (char *filename) {
  struct stat   buffer;   
  return (stat (filename, &buffer) == 0);
}


void print_dataset(){
   for(int i=0; i< TOTAL_OBSERVATIONS; i++){
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

   TOTAL_OBSERVATIONS = OBSERVATIONS * AUGMENT_FACTOR;

   dataset = (dataset_t *) malloc(sizeof(dataset_t)*TOTAL_OBSERVATIONS);

   if (file_exists(FILE_PATH_AUGMENTED)){
      printf("%s exists, loading file...\n", FILE_PATH_AUGMENTED);
      fp = fopen(FILE_PATH_AUGMENTED, "r");

   } else{
      printf("%s does not exit, loading file %s\n", FILE_PATH_AUGMENTED, FILE_PATH);
      fp = fopen(FILE_PATH, "r");
      TOTAL_OBSERVATIONS = OBSERVATIONS;
   }

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


void augment_dataset(){
   
   FILE *fp;
   char buffer[10000];
   char template[] = "%s_%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n";
   
   int *feature_set;

   fp = fopen(FILE_PATH_AUGMENTED, "w");

   for(int j=0; j<AUGMENT_FACTOR; j++){
      for(int i=0; i< OBSERVATIONS; i++){
         
         feature_set = dataset[i].features;
         
         sprintf(buffer, template, dataset[i].name, j, 
                                 feature_set[0],
                                 feature_set[1],
                                 feature_set[2],
                                 feature_set[3],
                                 feature_set[4],
                                 feature_set[5],
                                 feature_set[6],
                                 feature_set[7],
                                 feature_set[8],
                                 feature_set[9],
                                 feature_set[10],
                                 feature_set[11],
                                 feature_set[12],
                                 feature_set[13],
                                 feature_set[14],
                                 feature_set[15],
                                 dataset[i].class,
                                 dataset[i].label);
                                 
            fputs(buffer, fp);
         }
   }

   fclose(fp);

}


void free_dataset(){
   for(int i=0; i< OBSERVATIONS; i++){
      free(dataset[i].name);
   }

   free(dataset);

}
