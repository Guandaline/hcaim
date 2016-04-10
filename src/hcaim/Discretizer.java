package hcaim;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class Discretizer {

    private String in_train = "";
    private String out_train = "";
    private String in_test = "";
    private String out_test = "";

    public Discretizer(String[] args) {
        this.readParans(args);
    }

    private void readParans(String[] args) {
        for (int i = 0; i < args.length; i++) {
            String[] paran = args[i].split("--");

            switch (paran[0]) {
                case "-i":
                    this.in_train = paran[1];
                    break;
                case "-o":
                    this.out_train = paran[1];
                    break;
                case "-r":
                    this.in_test = paran[1];
                    break;
                case "-s":
                    this.out_test = paran[1];
                    break;
            }
        }
        
        if (this.out_train.isEmpty()) {
            this.out_train = this.in_train.replace(".arff", "_hcaim.arff");
        }         
        
        if (!this.in_test.isEmpty() && this.out_test.isEmpty()) {
            this.out_test = this.in_test.replace(".arff", "_hcaim.arff");
        }
    }

    private Instances loadBase(String file) throws IOException {
        FileReader reader = new FileReader(file);
        Instances base = new Instances(reader);
        reader.close();
        base.setClass(base.attribute("class"));
        return base;
    }

    private void saveBase(Instances base, String file) throws IOException {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(base);
        saver.setFile(new File(file));
        saver.writeBatch();
    }

    private void renameAttributeValues(Instances base) {
        for (int i = 0; i < base.numAttributes(); i++) {
            Attribute attr = base.attribute(i);

            for (int j = 0; j < attr.numValues(); j++) {

                if (i != base.classIndex()) {
                    base.renameAttributeValue(attr, attr.value(j), "" + j);
                }
            }
        }
    }
    
    private void saveResults(Instances data, Hierarchy discretizer, String file_name) throws IOException {
        Instances tes_result = discretizer.createData(data);
        tes_result.setClass(tes_result.attribute("class"));
        renameAttributeValues(tes_result);
        saveBase(tes_result, file_name);
    }

    public void run() throws IOException {
        Instances train = loadBase(this.in_train);

        Hierarchy discretizer = new HCaim(train);
        discretizer.run();

        this.saveResults(train, discretizer, this.out_train);

        if (!this.in_test.isEmpty()) {
            Instances teste = loadBase(this.in_test);
            this.saveResults(teste, discretizer, this.out_test);
        }
    }    
}
