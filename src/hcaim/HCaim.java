package hcaim;

import java.util.HashMap;
import java.util.Iterator;
import java.util.TreeMap;
import weka.core.Instance;
import weka.core.Instances;

public class HCaim extends Hierarchy {
    private TreeMap quantaMatrix;

    public HCaim(Instances data) {
        this.base = data;
    }

    @Override
    public TreeMap discretizeAttr(Double ini, Double fim) {
        TreeMap cut_points = (TreeMap) this.returnPotentialCutPoints(this.index_attr, ini, fim);
        TreeMap global_schema = new TreeMap<Double, Integer>();

        global_schema.put(ini, 1);
        global_schema.put(fim, 1);

        if (cut_points.size() <= this.base.numClasses()) {
            global_schema.putAll(cut_points);
            return global_schema;
        }
        
        this.createQuantaMatrix();
        return this.hCaim(global_schema, cut_points);
    }

    private TreeMap hCaim(TreeMap global_schema, TreeMap cut_points) {
        Boolean stop = true;
        Integer k = 1;
        Double global_caim = 0.0;
        Integer min_cut_points = this.base.numClasses();
        TreeMap local_schema = new TreeMap<Double, Integer>();
        this.intervals = new HashMap<String, Double>();

        do {
            Double local_caim = 0.0;
            Double caim = 0.0;
            Iterator it = cut_points.keySet().iterator();

            while (it.hasNext()) {

                TreeMap schema = new TreeMap<Double, Integer>();
                schema.putAll(global_schema);
                Double point = (Double) it.next();

                if (!schema.containsKey(point)) {
                    schema.put(point, 1);

                    caim = this.measure(schema, this.num_levels);

                    if (caim >= local_caim) {
                        local_caim = caim;
                        local_schema.clear();
                        local_schema.putAll(schema);
                    }
                }
            }

            stop = true;

            if (local_caim > global_caim || k < min_cut_points) {
                global_caim = local_caim;
                global_schema.clear();
                global_schema.putAll(local_schema);
                stop = false;
            }

            k++;


        } while (!stop);

        return global_schema;
    }

    public Double measure(TreeMap schema, Integer level) {
        Double caim = 0.0;
        Double left = null;
        Integer n_intervals = schema.size() - 1;
        Iterator it = schema.keySet().iterator();

        while (it.hasNext()) {

            Double right = (Double) it.next();

            if (left == null) {
                left = right;
                continue;
            }

            Double sum = 0.0;
            HashMap interval_classies = this.countClassIntervalQuantaMatrix(left, right, level);
            Integer depth = this.returnDeeperClassLevel(interval_classies);
            Double num_instances_interval = this.returnIntervalNumInstances(interval_classies).doubleValue();

            for (int l = 1; l <= depth; l++) {
                Double value = 0.0;
                String interval = l + "[" + left + ", " + right + "]";

                if (!this.intervals.containsKey(interval)) {
                    HashMap classies = this.countClassIntervalQuantaMatrix(left, right, l);
                    Double weight = this.returnWeight(l, depth);
                    Double max = this.returnMaxClassOccurrence(classies).doubleValue();
                    value = (Double) ((max / num_instances_interval) * max) * weight;
                    this.intervals.put(interval, value);
                } else {
                    value = (Double) this.intervals.get(interval);
                }

                sum = sum + value;
            }

            caim = caim + (sum.doubleValue());
            left = right;
        }

        return caim / n_intervals.doubleValue();
    }
    
    public HashMap countClassIntervalQuantaMatrix(Double begin, Double end, Integer level) {
        HashMap counts = new HashMap<Integer, HashMap>();
        TreeMap values = (TreeMap) this.quantaMatrix.get(level);
        Iterator it = values.keySet().iterator();

        while (it.hasNext()) {
            Double value = (Double) it.next();

            if (value > end) {
                return counts;
            }

            HashMap classes = (HashMap) values.get(value);

            if (begin < value && value <= end) {
                Iterator itc = classes.keySet().iterator();

                while (itc.hasNext()) {
                    String class_name = (String) itc.next();
                    class_name = this.getClassAtLevel(class_name, level);
                    Integer count = counts.containsKey(class_name) ? (Integer) counts.get(class_name) : 0;
                    counts.put(class_name, count + (Integer) classes.get(class_name));
                }
            }
        }

        return counts;
    }

    public void createQuantaMatrix() {
        this.quantaMatrix = new TreeMap<Integer, TreeMap>();

        for (int level = 1; level <= this.num_levels; level++) {
            TreeMap values = new TreeMap<Double, HashMap>();

            for (int j = 0; j < this.base.numInstances(); j++) {
                Instance instance = this.base.instance(j);
                Double value = instance.value(this.attr);

                String classValue = this.base.classAttribute().value((int) instance.classValue());
                String class_name = this.getClassAtLevel(classValue, level);
                HashMap classies = (HashMap) (values.containsKey(value) ? values.get(value) : new HashMap<String, Integer>());

                Integer count = classies.containsKey(class_name) ? (Integer) classies.get(class_name) : 0;
                classies.put(class_name, count + 1);
                values.put(value, classies);
            }

            this.quantaMatrix.put(level, values);
        }
    }
}
