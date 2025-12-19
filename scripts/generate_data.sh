for i in {00..20}; do
    for j in {0..3}; do
        for d in low mid high
        do 
            python -m generate_data.generate_data --dataset replica --scene v3_sc${j}_staging_${i} --semantic --obj_density $d
        done
    done
done