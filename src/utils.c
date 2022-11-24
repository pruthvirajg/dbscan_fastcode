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
   // for(int i=0; i< TOTAL_OBSERVATIONS; i++){
   //    printf("%d, %s, ", i, dataset[i].name);
      
   //    for(int j=0; j< FEATURES; j++){
   //       printf("%f, ", dataset[i].features[j]);
   //    }

   //    printf("%d, ", dataset[i].class);
   //    printf("%d", dataset[i].label);
   //    printf("\n");
   // }
   for(int i=0; i< TOTAL_OBSERVATIONS; i++){
      printf("%d, %s, ", i, name_arr[i]);
      
      for(int j=0; j< FEATURES; j++){
         printf("%f, ", features_arr[i][j]);
      }

      printf("%d, ", class_arr[i]);
      printf("%d", label_arr[i]);
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

   // dataset = (dataset_t *) malloc(sizeof(dataset_t)*TOTAL_OBSERVATIONS);
   name_arr = (char **) malloc(sizeof(char *) * TOTAL_OBSERVATIONS);
   features_arr = (float **) malloc(sizeof(float*) * TOTAL_OBSERVATIONS);
   class_arr = (int *) malloc(sizeof(int) * TOTAL_OBSERVATIONS);
   label_arr = (int *) malloc(sizeof(int) * TOTAL_OBSERVATIONS);

   if (file_exists(FILE_PATH_AUGMENTED)){
      printf("%s exists, loading file...\n", FILE_PATH_AUGMENTED);
      fp = fopen(FILE_PATH_AUGMENTED, "r");

   } else{
      printf("%s does not exist, loading file %s\n", FILE_PATH_AUGMENTED, FILE_PATH);
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
            // dataset[observation_count].name = (char *)malloc(sizeof(char)*strlen(ch));
            // strcpy(dataset[observation_count].name, ch);
            name_arr[observation_count] = (char *)malloc(sizeof(char)*strlen(ch));
            strcpy(name_arr[observation_count], ch);

            features_arr[observation_count] = (float *)malloc(sizeof(float) * FEATURES);
         }
         else if(struct_counter>=1 && struct_counter <= FEATURES){
            // dataset[observation_count].features[struct_counter - 1] = (DTYPE)atoi(ch);
            features_arr[observation_count][struct_counter - 1] = (DTYPE)atoi(ch);
         }
         else if(struct_counter == FEATURES + 1){
            // dataset[observation_count].class = atoi(ch);
            class_arr[observation_count] = atoi(ch);
         }
         else if(struct_counter == FEATURES + 2){
               // dataset[observation_count].label = ACC_DBSCAN ? NOISE : atoi(ch);
            label_arr[observation_count] = ACC_DBSCAN ? NOISE : atoi(ch);
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
   char template[] = "%s_%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d\n";
   
   DTYPE *feature_set;

   fp = fopen(FILE_PATH_AUGMENTED, "w");

   for(int j=0; j<AUGMENT_FACTOR; j++){
      for(int i=0; i< OBSERVATIONS; i++){
         
         // feature_set = dataset[i].features;
         feature_set = features_arr[i];

         // sprintf(buffer, template, dataset[i].name, j, 
         //                         feature_set[0],
         //                         feature_set[1],
         //                         feature_set[2],
         //                         feature_set[3],
         //                         feature_set[4],
         //                         feature_set[5],
         //                         feature_set[6],
         //                         feature_set[7],
         //                         feature_set[8],
         //                         feature_set[9],
         //                         feature_set[10],
         //                         feature_set[11],
         //                         feature_set[12],
         //                         feature_set[13],
         //                         feature_set[14],
         //                         feature_set[15],
         //                         dataset[i].class,
         //                         dataset[i].label);
         sprintf(buffer, template, name_arr[i], j, 
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
                                 class_arr[i],
                                 label_arr[i]);
                                 
            fputs(buffer, fp);
         }
   }

   fclose(fp);

}


void free_dataset(){
   // for(int i=0; i< OBSERVATIONS; i++){
   //    free(dataset[i].name);
   // }
   // free(dataset);

   for(int i=0; i < TOTAL_OBSERVATIONS; i++){
      free(name_arr[i]);
   }

   for(int i=0; i < TOTAL_OBSERVATIONS; i++){
      free(features_arr[i]);
   }

   free(class_arr);
   free(label_arr);
   free(name_arr);
   free(features_arr);
}
