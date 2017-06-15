for d in $(ls); do
    sed -i 's/\r//g' ${d}
done
