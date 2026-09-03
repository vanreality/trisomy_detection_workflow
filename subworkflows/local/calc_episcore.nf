//
// Calculate episcore using clean bam and deconvolution result
//

include { SAMTOOLS_INDEX as SAMTOOLS_INDEX_TARGET } from '../../modules/nf-core/samtools/index/main.nf'
include { SAMTOOLS_INDEX as SAMTOOLS_INDEX_BACKGROUND } from '../../modules/nf-core/samtools/index/main.nf'
include { METHYLDACKEL_EXTRACT as METHYLDACKEL_EXTRACT_TARGET } from '../../modules/nf-core/methyldackel/extract/main.nf'
include { METHYLDACKEL_EXTRACT as METHYLDACKEL_EXTRACT_BACKGROUND } from '../../modules/nf-core/methyldackel/extract/main.nf'
include { EXTRACT_BETA_VALUE } from '../../modules/local/extract_beta_value/main.nf'
include { BETA_TO_EPISCORE } from '../../modules/local/beta_to_episcore/main.nf'

workflow CALC_EPISCORE {
    take:
    ch_samplesheet // channel: samplesheet with columns ['sample', 'target_bam', 'background_bam']

    main:
    // Target bam processing
    ch_samplesheet.map { meta, target_bam, background_bam ->
        return [meta, target_bam]
    }.set { ch_target_bam }

    SAMTOOLS_INDEX_TARGET(ch_target_bam)
    ch_target_bam
        .join(SAMTOOLS_INDEX_TARGET.out.bai)
        .set { ch_target_bam_with_bai }

    ch_target_bam_with_bai
        .multiMap { meta, bam, bai ->
            bam_input: [ meta, bam ]
            bai_input: [ meta, bai ]
        }
        .set { ch_target_bam_with_bai_ready }

    METHYLDACKEL_EXTRACT_TARGET(
        ch_target_bam_with_bai_ready.bam_input,
        ch_target_bam_with_bai_ready.bai_input,
        [[:], file(params.fasta)],
        [[:], file(params.fasta_index)]
    )
    METHYLDACKEL_EXTRACT_TARGET.out.bedgraph
        .set { ch_target_bedgraph }

    // Background bam processing
    ch_samplesheet.map { meta, target_bam, background_bam ->
        return [meta, background_bam]
    }.set { ch_background_bam }

    SAMTOOLS_INDEX_BACKGROUND(ch_background_bam)
    ch_background_bam
        .join(SAMTOOLS_INDEX_BACKGROUND.out.bai)
        .set { ch_background_bam_with_bai }

    ch_background_bam_with_bai
        .multiMap { meta, bam, bai ->
            bam_input: [ meta, bam ]
            bai_input: [ meta, bai ]
        }
        .set { ch_background_bam_with_bai_ready }

    METHYLDACKEL_EXTRACT_BACKGROUND(
        ch_background_bam_with_bai_ready.bam_input,
        ch_background_bam_with_bai_ready.bai_input,
        [[:], file(params.fasta)],
        [[:], file(params.fasta_index)]
    )
    METHYLDACKEL_EXTRACT_BACKGROUND.out.bedgraph
        .set { ch_background_bedgraph }

    // Merge bedGraph
    ch_target_bedgraph
        .join(ch_background_bedgraph, by: 0)
        .map { meta, target_bedgraph, background_bedgraph ->
            return [meta, target_bedgraph, background_bedgraph]
        }
        .set { ch_merged_bedgraph }

    // Extract beta value
    EXTRACT_BETA_VALUE(
        ch_merged_bedgraph,
        file(params.cpg_list)
    )
    EXTRACT_BETA_VALUE.out.beta_value
        .set { ch_beta_value }

    // Calculate episcore
    def ref_matrix     = params.reference_episcore_matrix ? file(params.reference_episcore_matrix) : []
    
    BETA_TO_EPISCORE(
        ch_beta_value,
        ref_matrix,
        file(params.cpg_list),
        params.beta_depth_threshold
    )
    BETA_TO_EPISCORE.out.episcore
        .set { ch_episcore }

    emit:
    beta_value           = ch_beta_value
    episcore             = ch_episcore
    episcore_before_mq   = BETA_TO_EPISCORE.out.episcore_before_mq
}
